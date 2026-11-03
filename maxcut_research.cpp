// DATS-MaxCut : Degree-Aware Tiled Search for Max-Cut on GPUs
// (research version of our Parallel Computing mini project, VJTI Mumbai)
// Vedant Tamhanekar (241080072), Siddhant Salunke (241080062)
//
// maxcut.cpp only fixed the neighbour update part (degree aware warps / blocks).
// While profiling it we saw that the bigger cost is actually choosing the move:
// every step all n vertices are scanned to find the best gain, but one flip only
// changes gain of v and its neighbours. So in this version we add:
//
//   1. Tiled incremental selection: vertices are grouped in tiles of 32, every tile
//      keeps its best move cached. After a flip only the "dirty" tiles (tiles that
//      contain a neighbour of v) are recomputed, one warp per tile. Picking the best
//      move then needs only n/32 reads instead of n.
//   2. Degree aware neighbour update (from maxcut.cpp) - now it matters more, because
//      once selection is cheap the hub updates become the slow part.
//   3. BFS relabelling of vertices so neighbours get nearby ids and fall in the same
//      tiles, which means less dirty tiles per flip.
//
// The tiled selection is exact (gives the same move as a full scan), and all versions
// use the same random numbers, so CPU search i and GPU block i follow exactly the same
// path. We use this to check correctness and to compare speed fairly.
//
// compile (GPU + OpenMP):  nvcc -O3 -arch=sm_75 -x cu -Xcompiler -fopenmp -lgomp maxcut_research.cpp -o dats
// compile (CPU only):      g++ -O3 -fopenmp maxcut_research.cpp -o dats
//
// run:  ./dats -g data/G1.txt -name G1
//       ./dats -g data/email-Enron.txt -snap -name email-Enron -csv results.csv
// options: -steps T -blocks B -threads TPB -cpu S -seed X -only cpu|gpu -noserial -shuffle -gen n d

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

using namespace std;
typedef unsigned long long u64;

#define LOW_DEGREE_MAX 32
#define MEDIUM_DEGREE_MAX 1024
#define GROUP_LOW 0
#define GROUP_MEDIUM 1
#define GROUP_HIGH 2

// one tile = one warp worth of vertices. For the tile size experiment on CPU we
// compile with -DTILE_SIZE=16 etc, but the GPU code needs exactly 32 (lane = vertex)
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
#if defined(__CUDACC__) && TILE_SIZE != 32
#error "GPU version needs TILE_SIZE = 32"
#endif

// ----------------------------- small helpers -----------------------------

HOST_DEVICE inline unsigned int xorshift(unsigned int &state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

HOST_DEVICE inline unsigned int mixSeed(unsigned int a, unsigned int b)
{
    unsigned int h = a * 2654435761u ^ (b + 0x9E3779B9u + (a << 6) + (a >> 2));
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    return h | 1;
}

// A move is packed in one 64 bit "key" so comparing moves is just comparing numbers:
//   high 32 bits = gain (sign bit flipped so bigger gain = bigger unsigned number)
//   low 32 bits  = (v * HASH_MUL) ^ salt   -> random looking tie break between equal gains
// HASH_MUL is odd so it has an inverse mod 2^32, so we can get v back from the key.
// key 0 means "no move" (it would need gain = INT_MIN which never happens)
#define HASH_MUL 0x9E3779B1u
#define HASH_MUL_INV 0x0E8B2F51u

HOST_DEVICE inline u64 makeMoveKey(int gain, int v, unsigned int salt)
{
    unsigned int high = (unsigned int)gain ^ 0x80000000u;
    unsigned int low = ((unsigned int)v * HASH_MUL) ^ salt;
    return ((u64)high << 32) | low;
}

HOST_DEVICE inline int keyGain(u64 key)
{
    return (int)((unsigned int)(key >> 32) ^ 0x80000000u);
}

HOST_DEVICE inline int keyVertex(u64 key, unsigned int salt)
{
    return (int)((((unsigned int)key) ^ salt) * HASH_MUL_INV);
}

double secondsSince(chrono::high_resolution_clock::time_point start)
{
    return chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
}

// ----------------------------- graph part -----------------------------

struct Edge
{
    int u, v, w;
};

struct Graph
{
    int numVertices = 0;
    int numEdges = 0;
    vector<int> rowPtr;
    vector<int> colIdx;
    vector<int> edgeWeight;
    vector<unsigned char> degreeGroup;
};

void classifyDegrees(Graph &g)
{
    g.degreeGroup.resize(g.numVertices);
    for (int v = 0; v < g.numVertices; v++)
    {
        int d = g.rowPtr[v + 1] - g.rowPtr[v];
        g.degreeGroup[v] = d <= LOW_DEGREE_MAX ? GROUP_LOW : (d <= MEDIUM_DEGREE_MAX ? GROUP_MEDIUM : GROUP_HIGH);
    }
}

Graph buildCSR(int numVertices, vector<Edge> &edgeList)
{
    vector<Edge> cleanEdges;
    cleanEdges.reserve(edgeList.size());
    for (auto &e : edgeList)
    {
        if (e.u == e.v)
            continue;
        if (e.u > e.v)
            swap(e.u, e.v);
        cleanEdges.push_back(e);
    }
    sort(cleanEdges.begin(), cleanEdges.end(), [](const Edge &a, const Edge &b)
         { return a.u != b.u ? a.u < b.u : a.v < b.v; });

    vector<Edge> uniqueEdges;
    uniqueEdges.reserve(cleanEdges.size());
    for (size_t i = 0; i < cleanEdges.size(); i++)
    {
        if (i > 0 && cleanEdges[i].u == cleanEdges[i - 1].u && cleanEdges[i].v == cleanEdges[i - 1].v)
            continue;
        uniqueEdges.push_back(cleanEdges[i]);
    }

    Graph g;
    g.numVertices = numVertices;
    g.numEdges = (int)uniqueEdges.size();
    vector<int> degree(numVertices, 0);
    for (auto &e : uniqueEdges)
    {
        degree[e.u]++;
        degree[e.v]++;
    }
    g.rowPtr.assign(numVertices + 1, 0);
    for (int v = 0; v < numVertices; v++)
        g.rowPtr[v + 1] = g.rowPtr[v] + degree[v];
    g.colIdx.resize(g.rowPtr[numVertices]);
    g.edgeWeight.resize(g.rowPtr[numVertices]);
    vector<int> nextSlot(g.rowPtr.begin(), g.rowPtr.end() - 1);
    for (auto &e : uniqueEdges)
    {
        g.colIdx[nextSlot[e.u]] = e.v;
        g.edgeWeight[nextSlot[e.u]++] = e.w;
        g.colIdx[nextSlot[e.v]] = e.u;
        g.edgeWeight[nextSlot[e.v]++] = e.w;
    }
    classifyDegrees(g);
    return g;
}

bool readGsetFile(const string &fileName, Graph &g)
{
    ifstream fin(fileName);
    if (!fin.is_open())
        return false;
    int n, m;
    fin >> n >> m;
    vector<Edge> edges;
    edges.reserve(m);
    for (int i = 0; i < m; i++)
    {
        Edge e;
        if (!(fin >> e.u >> e.v >> e.w))
            break;
        e.u--;
        e.v--;
        edges.push_back(e);
    }
    g = buildCSR(n, edges);
    return true;
}

bool readSnapFile(const string &fileName, Graph &g)
{
    ifstream fin(fileName);
    if (!fin.is_open())
        return false;
    unordered_map<long long, int> idMap;
    vector<Edge> edges;
    string line;
    while (getline(fin, line))
    {
        if (line.empty() || line[0] == '#' || line[0] == '%')
            continue;
        stringstream ss(line);
        long long a, b;
        if (!(ss >> a >> b))
            continue;
        if (!idMap.count(a))
        {
            int newId = (int)idMap.size();
            idMap[a] = newId;
        }
        if (!idMap.count(b))
        {
            int newId = (int)idMap.size();
            idMap[b] = newId;
        }
        edges.push_back({idMap[a], idMap[b], 1});
    }
    g = buildCSR((int)idMap.size(), edges);
    return true;
}

// Chung-Lu power law graph (same as in maxcut.cpp)
Graph makePowerLawGraph(int numVertices, int avgDegree, unsigned int seed)
{
    const double gamma = 2.1;
    unsigned int rng = mixSeed(seed, 12345);
    vector<double> cumulativeWeight(numVertices);
    double totalWeight = 0;
    for (int i = 0; i < numVertices; i++)
    {
        totalWeight += pow(i + 1.0, -1.0 / (gamma - 1.0));
        cumulativeWeight[i] = totalWeight;
    }
    auto pickVertex = [&]()
    {
        double r = (xorshift(rng) / 4294967296.0) * totalWeight;
        int idx = (int)(upper_bound(cumulativeWeight.begin(), cumulativeWeight.end(), r) - cumulativeWeight.begin());
        return min(idx, numVertices - 1);
    };
    vector<Edge> edges;
    long long targetEdges = (long long)numVertices * avgDegree / 2;
    for (int v = 0; v < numVertices; v++)
    {
        int u = pickVertex();
        if (u == v)
            u = (v + 1) % numVertices;
        edges.push_back({v, u, 1});
    }
    while ((long long)edges.size() < targetEdges)
        edges.push_back({pickVertex(), pickVertex(), 1});
    return buildCSR(numVertices, edges);
}

// BFS relabelling. We start from the biggest hub, and new ids are given in BFS order,
// so all neighbours of a vertex that are discovered from it get consecutive ids
// and land in the same few tiles. Unreached parts start again from the next biggest vertex.
Graph reorderBFS(const Graph &g)
{
    int n = g.numVertices;
    vector<int> byDegree(n);
    for (int v = 0; v < n; v++)
        byDegree[v] = v;
    sort(byDegree.begin(), byDegree.end(), [&](int a, int b)
         {
             int da = g.rowPtr[a + 1] - g.rowPtr[a], db = g.rowPtr[b + 1] - g.rowPtr[b];
             return da != db ? da > db : a < b; });

    vector<int> newId(n, -1);
    vector<int> queue;
    queue.reserve(n);
    int nextLabel = 0;
    for (int root : byDegree)
    {
        if (newId[root] != -1)
            continue;
        size_t head = queue.size();
        queue.push_back(root);
        newId[root] = nextLabel++;
        while (head < queue.size())
        {
            int v = queue[head++];
            for (int e = g.rowPtr[v]; e < g.rowPtr[v + 1]; e++)
            {
                int u = g.colIdx[e];
                if (newId[u] == -1)
                {
                    newId[u] = nextLabel++;
                    queue.push_back(u);
                }
            }
        }
    }

    // build relabelled CSR directly (no need to sort again)
    Graph r;
    r.numVertices = n;
    r.numEdges = g.numEdges;
    vector<int> oldOf(n);
    for (int v = 0; v < n; v++)
        oldOf[newId[v]] = v;
    r.rowPtr.assign(n + 1, 0);
    for (int nv = 0; nv < n; nv++)
        r.rowPtr[nv + 1] = r.rowPtr[nv] + (g.rowPtr[oldOf[nv] + 1] - g.rowPtr[oldOf[nv]]);
    r.colIdx.resize(g.colIdx.size());
    r.edgeWeight.resize(g.edgeWeight.size());
    for (int nv = 0; nv < n; nv++)
    {
        int ov = oldOf[nv], pos = r.rowPtr[nv];
        for (int e = g.rowPtr[ov]; e < g.rowPtr[ov + 1]; e++, pos++)
        {
            r.colIdx[pos] = newId[g.colIdx[e]];
            r.edgeWeight[pos] = g.edgeWeight[e];
        }
    }
    classifyDegrees(r);
    return r;
}

// gives vertices random new ids, we use it to test what happens when the input file
// comes in a bad (random) order, since SNAP files are already in a fairly good order
Graph shuffleVertices(const Graph &g, unsigned int seed)
{
    int n = g.numVertices;
    vector<int> newId(n);
    for (int v = 0; v < n; v++)
        newId[v] = v;
    unsigned int rng = mixSeed(seed, 777);
    for (int i = n - 1; i > 0; i--)
        swap(newId[i], newId[xorshift(rng) % (i + 1)]);
    vector<Edge> edges;
    edges.reserve(g.numEdges);
    for (int v = 0; v < n; v++)
        for (int e = g.rowPtr[v]; e < g.rowPtr[v + 1]; e++)
            if (g.colIdx[e] > v)
                edges.push_back({newId[v], newId[g.colIdx[e]], g.edgeWeight[e]});
    return buildCSR(n, edges);
}

int maxDegree(const Graph &g)
{
    int best = 0;
    for (int v = 0; v < g.numVertices; v++)
        best = max(best, g.rowPtr[v + 1] - g.rowPtr[v]);
    return best;
}

void printGraphStats(const Graph &g)
{
    int n = g.numVertices;
    int groupCount[3] = {0, 0, 0};
    long long groupWork[3] = {0, 0, 0};
    for (int v = 0; v < n; v++)
    {
        groupCount[g.degreeGroup[v]]++;
        groupWork[g.degreeGroup[v]] += g.rowPtr[v + 1] - g.rowPtr[v];
    }
    long long totalAdj = g.rowPtr[n];
    printf("Graph: %d vertices, %d edges, max degree %d, avg degree %.2f\n", n, g.numEdges, maxDegree(g),
           n ? (double)totalAdj / n : 0.0);
    const char *groupName[3] = {"low   ", "medium", "high  "};
    for (int i = 0; i < 3; i++)
        printf("  %s : %7d vertices, %5.1f%% of adjacency entries\n", groupName[i], groupCount[i],
               totalAdj ? 100.0 * groupWork[i] / totalAdj : 0.0);
}

// average number of different tiles that the neighbours of a vertex fall in
// (this is roughly how many tiles get dirty when we flip it)
double averageTilesTouched(const Graph &g)
{
    long long total = 0;
    vector<int> lastSeen((g.numVertices + TILE_SIZE - 1) / TILE_SIZE, -1);
    for (int v = 0; v < g.numVertices; v++)
    {
        int count = 0;
        for (int e = g.rowPtr[v]; e < g.rowPtr[v + 1]; e++)
        {
            int t = g.colIdx[e] / TILE_SIZE;
            if (lastSeen[t] != v)
            {
                lastSeen[t] = v;
                count++;
            }
        }
        total += count;
    }
    return (double)total / max(1, g.numVertices);
}

long long computeCut(const Graph &g, const vector<char> &side)
{
    long long cut = 0;
    for (int v = 0; v < g.numVertices; v++)
        for (int e = g.rowPtr[v]; e < g.rowPtr[v + 1]; e++)
        {
            int u = g.colIdx[e];
            if (u > v && side[u] != side[v])
                cut += g.edgeWeight[e];
        }
    return cut;
}

// ----------------------------- search settings -----------------------------

struct SearchConfig
{
    int steps = 10000;
    int stallLimit = 500;
    int minPerturb = 2;
    int maxPerturb = 100;
    int tenureRange = 10;
    unsigned int baseSeed = 2024;
};

struct VersionResult
{
    string name;
    int numSearches = 0;
    long long bestCut = 0;
    double totalTime = 0;
    double timeToBest = 0;
    long long totalMoves = 0;
    double selectionWorkPerStep = -1; // only measured on CPU
    double dirtyTilesPerStep = -1;    // only measured on CPU
    double hubFlipPercent = -1;       // only measured on CPU
    bool verified = false;
    bool ran = false;
    vector<long long> perSearchCut; // used to check CPU and GPU give same answers
};

// ----------------------------- CPU tabu search -----------------------------

struct CpuSearchOutput
{
    long long bestCut = 0;
    double timeToBest = 0;
    long long moves = 0;
    long long selectionWork = 0; // number of vertex/tile entries read while choosing moves
    long long dirtyTiles = 0;
    long long hubFlips = 0;
    bool verified = false;
};

// One tabu search. With tiled=false it scans all vertices every step (normal way),
// with tiled=true it uses the cached tile maxima. Both pick exactly the same moves.
CpuSearchOutput tabuSearchCPU(const Graph &g, const SearchConfig &cfg, int searchId, bool tiled,
                              chrono::high_resolution_clock::time_point globalStart)
{
    CpuSearchOutput out;
    int n = g.numVertices;
    int numWords = (n + 31) / 32;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    // same random start as GPU block number searchId
    unsigned int searchSeed = mixSeed(cfg.baseSeed, searchId + 1);
    vector<char> side(n);
    for (int w = 0; w < numWords; w++)
    {
        unsigned int s = mixSeed(searchSeed, w);
        unsigned int bits = xorshift(s);
        for (int b = 0; b < 32 && w * 32 + b < n; b++)
            side[w * 32 + b] = (bits >> b) & 1;
    }
    unsigned int rng = searchSeed;

    vector<int> gain(n, 0);
    long long currentCut = 0;
    for (int v = 0; v < n; v++)
    {
        int gv = 0;
        for (int e = g.rowPtr[v]; e < g.rowPtr[v + 1]; e++)
        {
            int u = g.colIdx[e], w = g.edgeWeight[e];
            if (side[u] == side[v])
                gv += w;
            else
            {
                gv -= w;
                if (u > v)
                    currentCut += w;
            }
        }
        gain[v] = gv;
    }
    unsigned int salt = xorshift(rng); // tie break salt, changes after every perturbation

    vector<int> tabuUntil(n, 0); // v is tabu at step s if tabuUntil[v] > s

    // tile data
    vector<u64> tileFreeKey, tileTabuKey;
    vector<int> tileExpiry;
    vector<char> isDirty;
    vector<int> dirtyList;
    if (tiled)
    {
        tileFreeKey.assign(numTiles, 0);
        tileTabuKey.assign(numTiles, 0);
        tileExpiry.assign(numTiles, INT_MAX);
        isDirty.assign(numTiles, 0);
    }
    // recompute cached best of a tile, valid for steps >= forStep until its expiry
    auto refreshTile = [&](int t, int forStep)
    {
        u64 freeKey = 0, tabuKey = 0;
        int expiry = INT_MAX;
        int last = min(n, (t + 1) * TILE_SIZE);
        for (int v = t * TILE_SIZE; v < last; v++)
        {
            u64 key = makeMoveKey(gain[v], v, salt);
            if (tabuUntil[v] > forStep)
            {
                tabuKey = max(tabuKey, key);
                expiry = min(expiry, tabuUntil[v]);
            }
            else
                freeKey = max(freeKey, key);
        }
        tileFreeKey[t] = freeKey;
        tileTabuKey[t] = tabuKey;
        tileExpiry[t] = expiry;
    };
    auto markDirty = [&](int v)
    {
        int t = v / TILE_SIZE;
        if (!isDirty[t])
        {
            isDirty[t] = 1;
            dirtyList.push_back(t);
        }
    };
    auto rebuildAllTiles = [&](int forStep)
    {
        for (int t = 0; t < numTiles; t++)
            refreshTile(t, forStep);
        for (int t : dirtyList)
            isDirty[t] = 0;
        dirtyList.clear();
    };
    if (tiled)
        rebuildAllTiles(1);

    long long bestCut = currentCut;
    vector<char> bestSide = side;
    out.timeToBest = secondsSince(globalStart);
    int noImproveCount = 0;
    int perturbStrength = cfg.minPerturb;
    long long bestAtLastPerturb = bestCut;

    auto flipVertex = [&](int v)
    {
        for (int e = g.rowPtr[v]; e < g.rowPtr[v + 1]; e++)
        {
            int u = g.colIdx[e], w = g.edgeWeight[e];
            if (side[u] == side[v])
                gain[u] -= 2 * w;
            else
                gain[u] += 2 * w;
            if (tiled)
                markDirty(u);
        }
        currentCut += gain[v];
        gain[v] = -gain[v];
        side[v] ^= 1;
        if (tiled)
            markDirty(v);
    };
    auto checkBest = [&]()
    {
        if (currentCut > bestCut)
        {
            bestCut = currentCut;
            bestSide = side;
            out.timeToBest = secondsSince(globalStart);
            return true;
        }
        return false;
    };

    for (int step = 1; step <= cfg.steps; step++)
    {
        // ---- choose move ----
        u64 bestKey = 0;
        if (tiled)
        {
            for (int t = 0; t < numTiles; t++)
            {
                if (tileExpiry[t] <= step) // some tabu vertex in this tile became free
                {
                    refreshTile(t, step);
                    out.selectionWork += TILE_SIZE;
                }
                u64 candidate = tileFreeKey[t];
                u64 tk = tileTabuKey[t];
                if (tk && currentCut + keyGain(tk) > bestCut && tk > candidate) // aspiration
                    candidate = tk;
                bestKey = max(bestKey, candidate);
            }
            out.selectionWork += numTiles;
        }
        else
        {
            for (int v = 0; v < n; v++)
            {
                if (tabuUntil[v] > step && currentCut + gain[v] <= bestCut)
                    continue;
                bestKey = max(bestKey, makeMoveKey(gain[v], v, salt));
            }
            out.selectionWork += n;
        }
        int chosen = bestKey ? keyVertex(bestKey, salt) : (int)(xorshift(rng) % n);

        // ---- do move ----
        if (g.degreeGroup[chosen] == GROUP_HIGH)
            out.hubFlips++;
        flipVertex(chosen);
        tabuUntil[chosen] = step + 3 + xorshift(rng) % cfg.tenureRange;
        out.moves++;
        if (checkBest())
            noImproveCount = 0;
        else
            noImproveCount++;

        // ---- fix the tiles that changed ----
        if (tiled)
        {
            out.dirtyTiles += dirtyList.size();
            out.selectionWork += (long long)dirtyList.size() * TILE_SIZE;
            for (int t : dirtyList)
            {
                refreshTile(t, step + 1);
                isDirty[t] = 0;
            }
            dirtyList.clear();
        }

        // ---- perturbation ----
        if (noImproveCount >= cfg.stallLimit)
        {
            if (bestCut == bestAtLastPerturb)
                perturbStrength = min(perturbStrength + 1, cfg.maxPerturb);
            else
                perturbStrength = cfg.minPerturb;
            bestAtLastPerturb = bestCut;

            for (int k = 0; k < perturbStrength; k++)
                flipVertex(xorshift(rng) % n);
            out.moves += perturbStrength;
            checkBest();
            noImproveCount = 0;
            salt = xorshift(rng);
            if (tiled)
            {
                rebuildAllTiles(step + 1); // everything changed anyway
                out.selectionWork += (long long)numTiles * TILE_SIZE;
            }
        }
    }
    out.bestCut = bestCut;
    out.verified = (computeCut(g, bestSide) == bestCut);
    return out;
}

VersionResult runCPU(const Graph &g, const SearchConfig &cfg, int numSearches, bool tiled, bool useOpenMP,
                     const string &name)
{
    VersionResult result;
    result.name = name;
    result.numSearches = numSearches;
    vector<CpuSearchOutput> outputs(numSearches);

    auto start = chrono::high_resolution_clock::now();
    if (useOpenMP)
    {
#pragma omp parallel for schedule(dynamic, 1)
        for (int s = 0; s < numSearches; s++)
            outputs[s] = tabuSearchCPU(g, cfg, s, tiled, start);
    }
    else
    {
        for (int s = 0; s < numSearches; s++)
            outputs[s] = tabuSearchCPU(g, cfg, s, tiled, start);
    }
    result.totalTime = secondsSince(start);

    result.bestCut = LLONG_MIN;
    result.verified = true;
    long long work = 0, dirty = 0, hubs = 0, steps = 0;
    for (auto &o : outputs)
    {
        result.totalMoves += o.moves;
        result.perSearchCut.push_back(o.bestCut);
        result.verified = result.verified && o.verified;
        work += o.selectionWork;
        dirty += o.dirtyTiles;
        hubs += o.hubFlips;
        steps += cfg.steps;
        if (o.bestCut > result.bestCut || (o.bestCut == result.bestCut && o.timeToBest < result.timeToBest))
        {
            result.bestCut = o.bestCut;
            result.timeToBest = o.timeToBest;
        }
    }
    result.selectionWorkPerStep = (double)work / max(1LL, steps);
    result.dirtyTilesPerStep = tiled ? (double)dirty / max(1LL, steps) : -1;
    result.hubFlipPercent = 100.0 * hubs / max(1LL, steps);
    result.ran = true;
    return result;
}

// ----------------------------- GPU part -----------------------------
#ifdef __CUDACC__

#define CUDA_CHECK(call)                                                             \
    do                                                                               \
    {                                                                                \
        cudaError_t err = (call);                                                    \
        if (err != cudaSuccess)                                                      \
        {                                                                            \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

#define MODE_NAIVE 0
#define MODE_DEGREE_AWARE 1

struct GpuSearchArgs
{
    const int *rowPtr;
    const int *colIdx;
    const int *edgeWeight;
    const unsigned char *degreeGroup;
    int numVertices, numWords, numTiles, numDirtyWords;

    int steps, stallLimit, minPerturb, maxPerturb, tenureRange;
    unsigned int baseSeed;

    bool gainsInShared, tilesInShared;
    int *gainScratch; // global fallbacks, one slice per block
    int *tabuScratch;
    u64 *tileFreeScratch;
    u64 *tileTabuScratch;
    int *tileExpiryScratch;

    unsigned int *bestBitsOut;
    int *bestCutOut;
    int *bestStepOut;
    long long *movesOut;
};

// pointers + block level values that every device function needs
struct BlockState
{
    unsigned int *sideBits;
    int *gain;
    int *tabuUntil;
    u64 *tileFreeKey;
    u64 *tileTabuKey;
    int *tileExpiry;
    unsigned int *dirtyMask;
    int *currentCut;
    unsigned int *salt;
};

__device__ inline int getSide(const unsigned int *sideBits, int v)
{
    return (sideBits[v >> 5] >> (v & 31)) & 1;
}

template <bool TILED>
__device__ inline void updateNeighbourGain(int e, int sideOfV, const GpuSearchArgs &args, BlockState &st)
{
    int u = args.colIdx[e];
    int w = args.edgeWeight[e];
    if (getSide(st.sideBits, u) == sideOfV)
        st.gain[u] -= 2 * w;
    else
        st.gain[u] += 2 * w;
    if (TILED)
    {
        int t = u / TILE_SIZE;
        atomicOr(&st.dirtyMask[t >> 5], 1u << (t & 31));
    }
}

// whole warp recomputes one tile, lane i looks at vertex tile*32+i
__device__ void refreshTileWarp(int t, int forStep, const GpuSearchArgs &args, BlockState &st)
{
    int lane = threadIdx.x & 31;
    int v = t * TILE_SIZE + lane;
    u64 freeKey = 0, tabuKey = 0;
    int expiry = INT_MAX;
    if (v < args.numVertices)
    {
        u64 key = makeMoveKey(st.gain[v], v, *st.salt);
        int until = st.tabuUntil[v];
        if (until > forStep)
        {
            tabuKey = key;
            expiry = until;
        }
        else
            freeKey = key;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        freeKey = max(freeKey, __shfl_down_sync(0xffffffff, freeKey, offset));
        tabuKey = max(tabuKey, __shfl_down_sync(0xffffffff, tabuKey, offset));
        expiry = min(expiry, __shfl_down_sync(0xffffffff, expiry, offset));
    }
    if (lane == 0)
    {
        st.tileFreeKey[t] = freeKey;
        st.tileTabuKey[t] = tabuKey;
        st.tileExpiry[t] = expiry;
    }
}

// single thread version, used in selection when a tabu vertex expires (happens rarely)
__device__ void refreshTileSerial(int t, int forStep, const GpuSearchArgs &args, BlockState &st)
{
    u64 freeKey = 0, tabuKey = 0;
    int expiry = INT_MAX;
    int last = min(args.numVertices, (t + 1) * TILE_SIZE);
    for (int v = t * TILE_SIZE; v < last; v++)
    {
        u64 key = makeMoveKey(st.gain[v], v, *st.salt);
        int until = st.tabuUntil[v];
        if (until > forStep)
        {
            tabuKey = max(tabuKey, key);
            expiry = min(expiry, until);
        }
        else
            freeKey = max(freeKey, key);
    }
    st.tileFreeKey[t] = freeKey;
    st.tileTabuKey[t] = tabuKey;
    st.tileExpiry[t] = expiry;
}

// every warp takes some words of the dirty bitmask and refreshes the tiles set in it
__device__ void refreshDirtyTiles(int forStep, const GpuSearchArgs &args, BlockState &st)
{
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    int numWarps = blockDim.x >> 5;
    for (int w = warpId; w < args.numDirtyWords; w += numWarps)
    {
        unsigned int pending = st.dirtyMask[w];
        while (pending)
        {
            int b = __ffs(pending) - 1;
            pending &= pending - 1;
            refreshTileWarp(w * 32 + b, forStep, args, st);
        }
        __syncwarp();
        if (lane == 0)
            st.dirtyMask[w] = 0;
    }
}

__device__ void rebuildAllTiles(int forStep, const GpuSearchArgs &args, BlockState &st)
{
    int warpId = threadIdx.x >> 5;
    int numWarps = blockDim.x >> 5;
    for (int t = warpId; t < args.numTiles; t += numWarps)
        refreshTileWarp(t, forStep, args, st);
    for (int w = threadIdx.x; w < args.numDirtyWords; w += blockDim.x)
        st.dirtyMask[w] = 0;
}

// flip v, called by all threads. tabuStep > 0 means a normal move (v becomes tabu),
// tabuStep = 0 means perturbation move. rng is only touched by thread 0.
template <int UPDATE_MODE, bool TILED>
__device__ void flipVertexGPU(int v, int tabuStep, unsigned int *rng, const GpuSearchArgs &args, BlockState &st)
{
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warpId = tid >> 5;
    int edgeStart = args.rowPtr[v];
    int edgeEnd = args.rowPtr[v + 1];
    int degree = edgeEnd - edgeStart;
    int sideOfV = getSide(st.sideBits, v);

    if (UPDATE_MODE == MODE_NAIVE)
    {
        if (tid == v % (int)blockDim.x)
            for (int e = edgeStart; e < edgeEnd; e++)
                updateNeighbourGain<TILED>(e, sideOfV, args, st);
    }
    else
    {
        int group = args.degreeGroup[v];
        if (group == GROUP_LOW)
        {
            if (warpId == 0 && lane < degree)
                updateNeighbourGain<TILED>(edgeStart + lane, sideOfV, args, st);
        }
        else if (group == GROUP_MEDIUM)
        {
            if (warpId == 0)
                for (int e = edgeStart + lane; e < edgeEnd; e += 32)
                    updateNeighbourGain<TILED>(e, sideOfV, args, st);
        }
        else
        {
            for (int e = edgeStart + tid; e < edgeEnd; e += blockDim.x)
                updateNeighbourGain<TILED>(e, sideOfV, args, st);
        }
    }
    __syncthreads();

    if (tid == 0)
    {
        *st.currentCut += st.gain[v];
        st.gain[v] = -st.gain[v];
        st.sideBits[v >> 5] ^= (1u << (v & 31));
        if (tabuStep > 0)
            st.tabuUntil[v] = tabuStep + 3 + xorshift(*rng) % args.tenureRange;
        if (TILED)
        {
            int t = v / TILE_SIZE;
            st.dirtyMask[t >> 5] |= 1u << (t & 31);
        }
    }
    __syncthreads();
}

__device__ inline void saveBestSolution(const unsigned int *sideBits, const GpuSearchArgs &args)
{
    unsigned int *dst = args.bestBitsOut + (size_t)blockIdx.x * args.numWords;
    for (int i = threadIdx.x; i < args.numWords; i += blockDim.x)
        dst[i] = sideBits[i];
}

// max over the block using warp shuffles, result is valid in thread 0
__device__ u64 blockMaxKey(u64 myKey, u64 *warpScratch)
{
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    int numWarps = blockDim.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        myKey = max(myKey, __shfl_down_sync(0xffffffff, myKey, offset));
    if (lane == 0)
        warpScratch[warpId] = myKey;
    __syncthreads();
    if (warpId == 0)
    {
        myKey = lane < numWarps ? warpScratch[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1)
            myKey = max(myKey, __shfl_down_sync(0xffffffff, myKey, offset));
    }
    return myKey;
}

template <int UPDATE_MODE, bool TILED>
__global__ void tabuSearchKernel(GpuSearchArgs args)
{
    extern __shared__ u64 sharedPool[]; // u64 so the start is 8 byte aligned
    unsigned char *pool = (unsigned char *)sharedPool;

    int n = args.numVertices;
    int tid = threadIdx.x;

    // ---- carve the dynamic shared memory (same order as host side computes it) ----
    BlockState st;
    size_t offset = 0;
    if (TILED)
    {
        if (args.tilesInShared)
        {
            st.tileFreeKey = (u64 *)(pool + offset);
            offset += sizeof(u64) * args.numTiles;
            st.tileTabuKey = (u64 *)(pool + offset);
            offset += sizeof(u64) * args.numTiles;
            st.tileExpiry = (int *)(pool + offset);
            offset += sizeof(int) * args.numTiles;
        }
        else
        {
            st.tileFreeKey = args.tileFreeScratch + (size_t)blockIdx.x * args.numTiles;
            st.tileTabuKey = args.tileTabuScratch + (size_t)blockIdx.x * args.numTiles;
            st.tileExpiry = args.tileExpiryScratch + (size_t)blockIdx.x * args.numTiles;
        }
        st.dirtyMask = (unsigned int *)(pool + offset);
        offset += sizeof(unsigned int) * args.numDirtyWords;
    }
    st.sideBits = (unsigned int *)(pool + offset);
    offset += sizeof(unsigned int) * args.numWords;
    st.gain = args.gainsInShared ? (int *)(pool + offset) : args.gainScratch + (size_t)blockIdx.x * n;
    st.tabuUntil = args.tabuScratch + (size_t)blockIdx.x * n;

    __shared__ u64 warpScratch[32];
    __shared__ int chosenVertex;
    __shared__ int currentCut;
    __shared__ int bestCut;
    __shared__ int bestStep;
    __shared__ unsigned int blockRng;
    __shared__ unsigned int salt;
    __shared__ int noImproveCount;
    __shared__ int perturbStrength;
    __shared__ int bestAtLastPerturb;
    __shared__ bool foundBetter;
    __shared__ long long moveCount;
    st.currentCut = &currentCut;
    st.salt = &salt;

    // ---- random start (same numbers as CPU search blockIdx.x) ----
    unsigned int blockSeed = mixSeed(args.baseSeed, blockIdx.x + 1);
    for (int w = tid; w < args.numWords; w += blockDim.x)
    {
        unsigned int s = mixSeed(blockSeed, w);
        st.sideBits[w] = xorshift(s);
    }
    for (int v = tid; v < n; v += blockDim.x)
        st.tabuUntil[v] = 0;
    if (TILED)
        for (int w = tid; w < args.numDirtyWords; w += blockDim.x)
            st.dirtyMask[w] = 0;
    if (tid == 0)
    {
        currentCut = 0;
        blockRng = blockSeed;
        noImproveCount = 0;
        perturbStrength = args.minPerturb;
        moveCount = 0;
    }
    __syncthreads();

    int myCutPart = 0;
    for (int v = tid; v < n; v += blockDim.x)
    {
        int sideOfV = getSide(st.sideBits, v);
        int gv = 0;
        for (int e = args.rowPtr[v]; e < args.rowPtr[v + 1]; e++)
        {
            int u = args.colIdx[e], w = args.edgeWeight[e];
            if (getSide(st.sideBits, u) == sideOfV)
                gv += w;
            else
            {
                gv -= w;
                if (u > v)
                    myCutPart += w;
            }
        }
        st.gain[v] = gv;
    }
    atomicAdd(&currentCut, myCutPart);
    __syncthreads();
    if (tid == 0)
    {
        bestCut = currentCut;
        bestAtLastPerturb = currentCut;
        bestStep = 0;
        salt = xorshift(blockRng);
    }
    __syncthreads();
    if (TILED)
        rebuildAllTiles(1, args, st);
    saveBestSolution(st.sideBits, args);
    __syncthreads();

    for (int step = 1; step <= args.steps; step++)
    {
        // ---- choose move ----
        int cutNow = currentCut;
        int bestNow = bestCut;
        unsigned int saltNow = salt;
        u64 myKey = 0;
        if (TILED)
        {
            for (int t = tid; t < args.numTiles; t += blockDim.x)
            {
                if (st.tileExpiry[t] <= step)
                    refreshTileSerial(t, step, args, st);
                u64 candidate = st.tileFreeKey[t];
                u64 tk = st.tileTabuKey[t];
                if (tk && cutNow + keyGain(tk) > bestNow && tk > candidate)
                    candidate = tk;
                myKey = max(myKey, candidate);
            }
        }
        else
        {
            for (int v = tid; v < n; v += blockDim.x)
            {
                int g = st.gain[v];
                if (st.tabuUntil[v] > step && cutNow + g <= bestNow)
                    continue;
                myKey = max(myKey, makeMoveKey(g, v, saltNow));
            }
        }
        myKey = blockMaxKey(myKey, warpScratch);
        if (tid == 0)
            chosenVertex = myKey ? keyVertex(myKey, saltNow) : (int)(xorshift(blockRng) % n);
        __syncthreads();

        // ---- do move ----
        flipVertexGPU<UPDATE_MODE, TILED>(chosenVertex, step, &blockRng, args, st);
        if (tid == 0)
        {
            moveCount++;
            foundBetter = false;
            if (currentCut > bestCut)
            {
                bestCut = currentCut;
                bestStep = step;
                noImproveCount = 0;
                foundBetter = true;
            }
            else
                noImproveCount++;
        }
        if (TILED)
            refreshDirtyTiles(step + 1, args, st); // tid 0 does not touch tiles above, so no sync needed before
        __syncthreads();
        if (foundBetter)
            saveBestSolution(st.sideBits, args);

        // ---- perturbation ----
        if (noImproveCount >= args.stallLimit)
        {
            if (tid == 0)
            {
                if (bestCut == bestAtLastPerturb)
                    perturbStrength = min(perturbStrength + 1, args.maxPerturb);
                else
                    perturbStrength = args.minPerturb;
                bestAtLastPerturb = bestCut;
            }
            __syncthreads();
            int strength = perturbStrength;
            for (int k = 0; k < strength; k++)
            {
                if (tid == 0)
                    chosenVertex = xorshift(blockRng) % n;
                __syncthreads();
                flipVertexGPU<UPDATE_MODE, TILED>(chosenVertex, 0, &blockRng, args, st);
            }
            if (tid == 0)
            {
                moveCount += strength;
                noImproveCount = 0;
                foundBetter = false;
                if (currentCut > bestCut)
                {
                    bestCut = currentCut;
                    bestStep = step;
                    foundBetter = true;
                }
                salt = xorshift(blockRng);
            }
            __syncthreads();
            if (foundBetter)
                saveBestSolution(st.sideBits, args);
            if (TILED)
                rebuildAllTiles(step + 1, args, st);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        args.bestCutOut[blockIdx.x] = bestCut;
        args.bestStepOut[blockIdx.x] = bestStep;
        args.movesOut[blockIdx.x] = moveCount;
    }
}

struct DeviceGraph
{
    int *rowPtr = nullptr;
    int *colIdx = nullptr;
    int *edgeWeight = nullptr;
    unsigned char *degreeGroup = nullptr;
};

DeviceGraph copyGraphToGPU(const Graph &g)
{
    DeviceGraph dg;
    size_t adjSize = g.colIdx.size();
    CUDA_CHECK(cudaMalloc(&dg.rowPtr, (g.numVertices + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dg.colIdx, max<size_t>(adjSize, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dg.edgeWeight, max<size_t>(adjSize, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dg.degreeGroup, g.numVertices));
    CUDA_CHECK(cudaMemcpy(dg.rowPtr, g.rowPtr.data(), (g.numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg.colIdx, g.colIdx.data(), adjSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg.edgeWeight, g.edgeWeight.data(), adjSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg.degreeGroup, g.degreeGroup.data(), g.numVertices, cudaMemcpyHostToDevice));
    return dg;
}

void freeDeviceGraph(DeviceGraph &dg)
{
    cudaFree(dg.rowPtr);
    cudaFree(dg.colIdx);
    cudaFree(dg.edgeWeight);
    cudaFree(dg.degreeGroup);
}

template <int UPDATE_MODE, bool TILED>
void launchKernel(int blocks, int threads, size_t sharedBytes, const GpuSearchArgs &args)
{
    CUDA_CHECK(cudaFuncSetAttribute(tabuSearchKernel<UPDATE_MODE, TILED>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes));
    tabuSearchKernel<UPDATE_MODE, TILED><<<blocks, threads, sharedBytes>>>(args);
}

VersionResult runGPU(const Graph &g, const DeviceGraph &dg, const SearchConfig &cfg, int numBlocks,
                     int threadsPerBlock, int mode, bool tiled, const string &name, bool quiet = false)
{
    VersionResult result;
    result.name = name;
    result.numSearches = numBlocks;

    int n = g.numVertices;
    GpuSearchArgs args;
    memset(&args, 0, sizeof(args));
    args.rowPtr = dg.rowPtr;
    args.colIdx = dg.colIdx;
    args.edgeWeight = dg.edgeWeight;
    args.degreeGroup = dg.degreeGroup;
    args.numVertices = n;
    args.numWords = (n + 31) / 32;
    args.numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    args.numDirtyWords = (args.numTiles + 31) / 32;
    args.steps = cfg.steps;
    args.stallLimit = cfg.stallLimit;
    args.minPerturb = cfg.minPerturb;
    args.maxPerturb = cfg.maxPerturb;
    args.tenureRange = cfg.tenureRange;
    args.baseSeed = cfg.baseSeed;

    // ---- decide what goes in shared memory: bits first, then tiles, then gains ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t sharedLimit = max(prop.sharedMemPerBlockOptin, prop.sharedMemPerBlock) - 2048;
    size_t bitsBytes = sizeof(unsigned int) * args.numWords;
    size_t dirtyBytes = tiled ? sizeof(unsigned int) * args.numDirtyWords : 0;
    size_t tileBytes = tiled ? (2 * sizeof(u64) + sizeof(int)) * args.numTiles : 0;
    size_t gainBytes = sizeof(int) * (size_t)n;
    if (bitsBytes + dirtyBytes > sharedLimit)
    {
        printf("[%s] graph too big for shared memory, skipped\n", name.c_str());
        return result;
    }
    size_t used = bitsBytes + dirtyBytes;
    args.tilesInShared = tiled && used + tileBytes <= sharedLimit;
    if (args.tilesInShared)
        used += tileBytes;
    args.gainsInShared = used + gainBytes <= sharedLimit;
    if (args.gainsInShared)
        used += gainBytes;
    size_t sharedBytes = used;

    if (!args.gainsInShared)
        CUDA_CHECK(cudaMalloc(&args.gainScratch, (size_t)numBlocks * n * sizeof(int)));
    if (tiled && !args.tilesInShared)
    {
        CUDA_CHECK(cudaMalloc(&args.tileFreeScratch, (size_t)numBlocks * args.numTiles * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&args.tileTabuScratch, (size_t)numBlocks * args.numTiles * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&args.tileExpiryScratch, (size_t)numBlocks * args.numTiles * sizeof(int)));
    }
    CUDA_CHECK(cudaMalloc(&args.tabuScratch, (size_t)numBlocks * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&args.bestBitsOut, (size_t)numBlocks * args.numWords * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&args.bestCutOut, numBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&args.bestStepOut, numBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&args.movesOut, numBlocks * sizeof(long long)));

    if (!quiet)
        printf("[%s] shared=%zu B, tiles in %s, gains in %s\n", name.c_str(), sharedBytes,
               tiled ? (args.tilesInShared ? "shared" : "global") : "-", args.gainsInShared ? "shared" : "global");

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));
    if (mode == MODE_NAIVE && !tiled)
        launchKernel<MODE_NAIVE, false>(numBlocks, threadsPerBlock, sharedBytes, args);
    else if (mode == MODE_NAIVE && tiled)
        launchKernel<MODE_NAIVE, true>(numBlocks, threadsPerBlock, sharedBytes, args);
    else if (!tiled)
        launchKernel<MODE_DEGREE_AWARE, false>(numBlocks, threadsPerBlock, sharedBytes, args);
    else
        launchKernel<MODE_DEGREE_AWARE, true>(numBlocks, threadsPerBlock, sharedBytes, args);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float kernelMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, startEvent, stopEvent));

    vector<int> blockBestCut(numBlocks), blockBestStep(numBlocks);
    vector<long long> blockMoves(numBlocks);
    CUDA_CHECK(cudaMemcpy(blockBestCut.data(), args.bestCutOut, numBlocks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(blockBestStep.data(), args.bestStepOut, numBlocks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(blockMoves.data(), args.movesOut, numBlocks * sizeof(long long), cudaMemcpyDeviceToHost));

    int winner = 0;
    for (int b = 0; b < numBlocks; b++)
    {
        result.totalMoves += blockMoves[b];
        result.perSearchCut.push_back(blockBestCut[b]);
        if (blockBestCut[b] > blockBestCut[winner] ||
            (blockBestCut[b] == blockBestCut[winner] && blockBestStep[b] < blockBestStep[winner]))
            winner = b;
    }
    result.bestCut = blockBestCut[winner];
    result.totalTime = kernelMs / 1000.0;
    result.timeToBest = result.totalTime * blockBestStep[winner] / max(1, cfg.steps); // estimate
    result.ran = true;

    vector<unsigned int> winnerBits(args.numWords);
    CUDA_CHECK(cudaMemcpy(winnerBits.data(), args.bestBitsOut + (size_t)winner * args.numWords,
                          args.numWords * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    vector<char> side(n);
    for (int v = 0; v < n; v++)
        side[v] = (winnerBits[v >> 5] >> (v & 31)) & 1;
    result.verified = (computeCut(g, side) == result.bestCut);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(args.gainScratch);
    cudaFree(args.tileFreeScratch);
    cudaFree(args.tileTabuScratch);
    cudaFree(args.tileExpiryScratch);
    cudaFree(args.tabuScratch);
    cudaFree(args.bestBitsOut);
    cudaFree(args.bestCutOut);
    cudaFree(args.bestStepOut);
    cudaFree(args.movesOut);
    return result;
}

#endif // __CUDACC__

// ----------------------------- main -----------------------------

// checks that two versions found the same best cut for every search they both ran
bool sameTrajectories(const VersionResult &a, const VersionResult &b)
{
    size_t common = min(a.perSearchCut.size(), b.perSearchCut.size());
    for (size_t i = 0; i < common; i++)
        if (a.perSearchCut[i] != b.perSearchCut[i])
            return false;
    return common > 0;
}

void printUsage(const char *prog)
{
    printf("Usage: %s (-g <file> [-snap] | -gen <n> <avgdeg>) [-name label] [-csv out.csv]\n", prog);
    printf("       [-steps T] [-blocks B] [-threads TPB] [-cpu S] [-seed X] [-only cpu|gpu] [-noserial] [-shuffle] [-brief] [-dats]\n\n");
}

int main(int argc, char **argv)
{
    string graphFile = "", label = "", csvFile = "", only = "";
    bool isSnap = false, noSerial = false, shuffle = false, brief = false, datsOnly = false;
    int genVertices = 0, genAvgDegree = 0;
    int numBlocks = 0; // 0 = decide from number of SMs
    int threadsPerBlock = 256;
    int cpuSearches = 8;
    SearchConfig cfg;

    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-g" && i + 1 < argc)
            graphFile = argv[++i];
        else if (arg == "-snap")
            isSnap = true;
        else if (arg == "-gen" && i + 2 < argc)
        {
            genVertices = atoi(argv[++i]);
            genAvgDegree = atoi(argv[++i]);
        }
        else if (arg == "-name" && i + 1 < argc)
            label = argv[++i];
        else if (arg == "-csv" && i + 1 < argc)
            csvFile = argv[++i];
        else if (arg == "-only" && i + 1 < argc)
            only = argv[++i];
        else if (arg == "-steps" && i + 1 < argc)
            cfg.steps = atoi(argv[++i]);
        else if (arg == "-blocks" && i + 1 < argc)
            numBlocks = atoi(argv[++i]);
        else if (arg == "-threads" && i + 1 < argc)
            threadsPerBlock = atoi(argv[++i]);
        else if (arg == "-cpu" && i + 1 < argc)
            cpuSearches = atoi(argv[++i]);
        else if (arg == "-seed" && i + 1 < argc)
            cfg.baseSeed = (unsigned int)atoi(argv[++i]);
        else if (arg == "-noserial")
            noSerial = true;
        else if (arg == "-shuffle")
            shuffle = true;
        else if (arg == "-brief")
            brief = true;
        else if (arg == "-dats")
            datsOnly = true; // print only our proposed method (gpu_dats), speedup is vs cpu_full
        else
        {
            printUsage(argv[0]);
            return 1;
        }
    }
    if (threadsPerBlock < 32 || threadsPerBlock > 1024 || threadsPerBlock % 32 != 0)
    {
        printf("threads per block should be a multiple of 32 between 32 and 1024\n");
        return 1;
    }

    Graph g;
    if (!graphFile.empty())
    {
        bool ok = isSnap ? readSnapFile(graphFile, g) : readGsetFile(graphFile, g);
        if (!ok)
        {
            printf("Could not open file %s\n", graphFile.c_str());
            return 1;
        }
        if (label.empty())
            label = graphFile;
    }
    else
    {
        if (genVertices == 0)
        {
            genVertices = 20000;
            genAvgDegree = 8;
        }
        g = makePowerLawGraph(genVertices, max(2, genAvgDegree), cfg.baseSeed);
        if (label.empty())
            label = "powerlaw-" + to_string(genVertices);
    }
    if (g.numVertices < 2)
    {
        printf("Graph is too small\n");
        return 1;
    }
    if (shuffle)
    {
        g = shuffleVertices(g, cfg.baseSeed);
        label += "-shuffled";
    }
    Graph reordered = reorderBFS(g);

    printf("=== %s ===\n", label.c_str());
    if (datsOnly)
        printf("Graph: %d vertices, %d edges, max degree %d\n", g.numVertices, g.numEdges, maxDegree(g));
    else
    {
        printGraphStats(g);
        printf("Avg tiles touched per vertex: original order %.2f, BFS order %.2f\n", averageTilesTouched(g),
               averageTilesTouched(reordered));
    }

    int n = g.numVertices;
    cfg.tenureRange = max(1, n / 10);
    cfg.minPerturb = min(max(2, n / 100), 100);
    cfg.maxPerturb = min(max(cfg.minPerturb + 1, n / 20), 500);
    if (!datsOnly)
        printf("steps=%d stall=%d perturb=%d..%d tenure=3..%d\n\n", cfg.steps, cfg.stallLimit, cfg.minPerturb,
               cfg.maxPerturb, cfg.tenureRange + 2);

    vector<VersionResult> results;

    if (only != "gpu")
    {
        if (!noSerial)
        {
            if (!datsOnly)
                printf("running CPU versions (serial)...\n");
            results.push_back(runCPU(g, cfg, cpuSearches, false, false, "cpu_full"));
            results.push_back(runCPU(g, cfg, cpuSearches, true, false, "cpu_tiled"));
            results.push_back(runCPU(reordered, cfg, cpuSearches, true, false, "cpu_tiled_bfs"));
            if (!datsOnly)
                printf("  tiled selection gives same result as full scan: %s\n",
                       sameTrajectories(results[0], results[1]) ? "YES" : "NO (bug!)");
        }
        if (!datsOnly)
            printf("running CPU versions (OpenMP)...\n");
        results.push_back(runCPU(g, cfg, cpuSearches, false, true, "cpu_full_omp"));
        results.push_back(runCPU(reordered, cfg, cpuSearches, true, true, "cpu_tiled_bfs_omp"));
    }

#ifdef __CUDACC__
    int deviceCount = 0;
    if (only != "cpu" && cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0)
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (numBlocks == 0)
            numBlocks = 2 * prop.multiProcessorCount;
        printf("\nGPU: %s, %d SMs, %d blocks x %d threads\n", prop.name, prop.multiProcessorCount, numBlocks,
               threadsPerBlock);

        DeviceGraph dgOriginal = copyGraphToGPU(g);
        DeviceGraph dgReordered = copyGraphToGPU(reordered);

        SearchConfig warmup = cfg;
        warmup.steps = 10;
        runGPU(g, dgOriginal, warmup, 1, threadsPerBlock, MODE_DEGREE_AWARE, true, "warmup", true);

        size_t firstGpu = results.size();
        results.push_back(runGPU(g, dgOriginal, cfg, numBlocks, threadsPerBlock, MODE_NAIVE, false, "gpu_naive_full", datsOnly));
        results.push_back(runGPU(g, dgOriginal, cfg, numBlocks, threadsPerBlock, MODE_DEGREE_AWARE, false, "gpu_degree_full", datsOnly));
        results.push_back(runGPU(g, dgOriginal, cfg, numBlocks, threadsPerBlock, MODE_DEGREE_AWARE, true, "gpu_degree_tiled", datsOnly));
        results.push_back(runGPU(reordered, dgReordered, cfg, numBlocks, threadsPerBlock, MODE_NAIVE, true, "gpu_naive_tiled_bfs", datsOnly));
        results.push_back(runGPU(reordered, dgReordered, cfg, numBlocks, threadsPerBlock, MODE_DEGREE_AWARE, true, "gpu_dats"));

        // all GPU versions on the same graph order should follow the same path as each other and as the CPU
        bool allSame = sameTrajectories(results[firstGpu], results[firstGpu + 1]) &&
                       sameTrajectories(results[firstGpu], results[firstGpu + 2]) &&
                       sameTrajectories(results[firstGpu + 3], results[firstGpu + 4]);
        printf("  all GPU variants follow identical search paths: %s\n", allSame ? "YES" : "NO (bug!)");
        for (auto &r : results)
            if (r.name == "cpu_full_omp")
                printf("  GPU block i == CPU search i: %s\n",
                       sameTrajectories(r, results[firstGpu]) ? "YES" : "NO (bug!)");

        freeDeviceGraph(dgOriginal);
        freeDeviceGraph(dgReordered);
    }
#else
    (void)numBlocks;
    if (only != "cpu")
        printf("\n(compiled without CUDA, GPU versions skipped)\n");
#endif

    // ---- table ----
    double baseRate = 0;
    for (auto &r : results)
        if (r.ran && r.name == "cpu_full")
            baseRate = r.totalMoves / r.totalTime;
    printf("\n%-20s %8s %9s %9s %13s %9s %10s %8s %6s %4s\n", "version", "searches", "best cut", "time(s)",
           "moves/s", "speedup", "sel.work", "dirty", "hub%", "ok");
    for (auto &r : results)
    {
        if (!r.ran)
            continue;
        // -brief: only serial CPU, naive GPU and our GPU version
        if (brief && r.name != "cpu_full" && r.name != "gpu_naive_full" && r.name != "gpu_dats")
            continue;
        if (datsOnly && r.name != "gpu_dats")
            continue;
        double rate = r.totalMoves / max(r.totalTime, 1e-9);
        printf("%-20s %8d %9lld %9.3f %13.0f ", r.name.c_str(), r.numSearches, r.bestCut, r.totalTime, rate);
        if (baseRate > 0)
            printf("%8.2fx ", rate / baseRate);
        else
            printf("%9s ", "-");
        if (r.selectionWorkPerStep >= 0)
            printf("%10.1f ", r.selectionWorkPerStep);
        else
            printf("%10s ", "-");
        if (r.dirtyTilesPerStep >= 0)
            printf("%8.2f ", r.dirtyTilesPerStep);
        else
            printf("%8s ", "-");
        if (r.hubFlipPercent >= 0)
            printf("%6.2f ", r.hubFlipPercent);
        else
            printf("%6s ", "-");
        printf("%4s\n", r.verified ? "yes" : "NO");
    }

    if (!csvFile.empty())
    {
        FILE *f = fopen(csvFile.c_str(), "a");
        if (f)
        {
            fseek(f, 0, SEEK_END);
            if (ftell(f) == 0)
                fprintf(f, "graph,n,m,maxdeg,version,searches,steps,best_cut,time_s,moves,moves_per_s,"
                           "time_to_best_s,sel_work_per_step,dirty_tiles_per_step,hub_flip_pct,"
                           "tiles_touched_orig,tiles_touched_bfs,verified\n");
            double touchedOrig = averageTilesTouched(g), touchedBfs = averageTilesTouched(reordered);
            for (auto &r : results)
                if (r.ran)
                    fprintf(f, "%s,%d,%d,%d,%s,%d,%d,%lld,%.6f,%lld,%.1f,%.6f,%.2f,%.3f,%.3f,%.3f,%.3f,%d\n",
                            label.c_str(), n, g.numEdges, maxDegree(g), r.name.c_str(), r.numSearches, cfg.steps,
                            r.bestCut, r.totalTime, r.totalMoves, r.totalMoves / max(r.totalTime, 1e-9),
                            r.timeToBest, r.selectionWorkPerStep, r.dirtyTilesPerStep, r.hubFlipPercent,
                            touchedOrig, touchedBfs, r.verified ? 1 : 0);
            fclose(f);
        }
    }
    return 0;
}
