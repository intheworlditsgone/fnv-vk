// Record thread (included in Backend::Impl; docs/PERFORMANCE.md "Record thread"). With BackendConfig::recordThread the
// public calls are queued as commands and run in order on a worker thread, which records them into Vulkan; the caller
// (the game's main thread, and its loader threads for textures) only copies the call's arguments.
//
// - A command is a header, a callable and its payload (copies of what the call points to), placed in a chunk of
//   memory; chunks go to the worker every kPublishEvery commands and at every present and wait.
// - Calls that return what only the recording knows wait for the queue to drain (sync): the worker runs the call and
//   signals. Calls whose results are not used, or are known to the caller, are queued (post).
// - present() waits until the previous present has run: the caller is at most one frame ahead.
// - A draw's render states, sampler states and float constants are pointers into the caller's tracker that change
//   after the call: the command carries what changed since the previous draw (render states when their version
//   changes, sampler states when they differ, and of the constants only the registers each shader reads,
//   PackedShader::constRuns, when they differ); the worker keeps the arrays they apply to.
// - The sun-lighting gate (SunLighting::gate) reads the game's current render pass, so the caller evaluates it when the
//   draw is queued and the worker uses that answer.
// - Occlusion query results are published by the worker when the GPU has them (checked after every chunk); the caller
//   reads the published result of a query's latest end without waiting.

struct CmdHeader {
  void (*run)(uint8_t* cmd);
  uint32_t size;      // the whole command, aligned
  uint32_t payload;   // offset of the payload from the header
};
static constexpr size_t kCmdAlign = 16;
static constexpr size_t cmdAlign(size_t n) { return (n + kCmdAlign - 1) & ~(kCmdAlign - 1); }
static constexpr size_t kChunkBytes = 1u << 20;
static constexpr uint32_t kPublishEvery = 64;

struct CmdChunk {
  uint8_t* data = nullptr;
  size_t cap = 0, used = 0;
  explicit CmdChunk(size_t bytes) : data(static_cast<uint8_t*>(_aligned_malloc(bytes, kCmdAlign))), cap(bytes) {}
  ~CmdChunk() { _aligned_free(data); }
  CmdChunk(const CmdChunk&) = delete;
  CmdChunk& operator=(const CmdChunk&) = delete;
};

struct Recorder {
  Impl& m;
  explicit Recorder(Impl& impl) : m(impl) {}

  // --- time (BackendStats::record*) ------------------------------------------------------------------------------
  std::atomic<uint64_t> busyUs{ 0 }, presentWaitUs{ 0 }, syncWaitUs{ 0 }, syncs{ 0 };

  // --- queued texture uploads (Backend::createTexture) -------------------------------------------------------------
  // A queued upload carries the decoded texture, which lives in the 32-bit address space until the worker has copied it
  // to staging. Loading a save decodes hundreds of textures on the game's loader threads faster than they are uploaded,
  // so the bytes queued are bounded: a caller that would pass kTexQueueBudget first waits for the queue to drain.
  // texQueuedPeak and texWaits: BackendStats::textureQueue*.
  static constexpr uint64_t kTexQueueBudget = 64ull << 20;
  std::atomic<uint64_t> texQueued{ 0 }, texQueuedPeak{ 0 }, texWaits{ 0 };
  static uint64_t nowUs() {
    static const double perUs = [] {
      LARGE_INTEGER f;
      QueryPerformanceFrequency(&f);
      return double(f.QuadPart) / 1e6;
    }();
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return uint64_t(double(t.QuadPart) / perUs);
  }

  // --- queue -----------------------------------------------------------------------------------------------------
  std::mutex pmu;                  // producers (the main thread, loader threads)
  CmdChunk* cur = nullptr;         // pmu
  uint32_t unpublished = 0;        // pmu
  std::mutex qmu;
  std::condition_variable qcv;
  std::deque<CmdChunk*> ready;     // qmu
  std::vector<CmdChunk*> spare;    // qmu
  bool stopping = false;           // qmu
  std::thread worker;
  std::mutex dmu;                  // completion
  std::condition_variable dcv;
  uint64_t syncIssued = 0, presentsIssued = 0;  // pmu
  uint64_t syncDone = 0, presentsDone = 0;      // dmu

  // Bytes of chunks holding commands not yet run (taken for the queue, until the worker has run them), bounded by
  // kQueueBudget: the copies a call carries (buffer data above all) live in the 32-bit address space, and loader threads
  // streaming a cell can queue faster than the worker records. A producer past the budget waits for the worker
  // (BackendStats/MemoryReport queue*). The worker never takes pmu, so waiting with it held cannot deadlock.
  static constexpr uint64_t kQueueBudget = 96ull << 20;
  std::condition_variable freedcv;                          // qmu: queued fell
  std::atomic<uint64_t> queued{ 0 }, queuedPeak{ 0 }, queueWaits{ 0 }, chunkAllocFailures{ 0 };  // queued: under qmu

  // A chunk with room for `bytes` (pmu held): the current one, a spare, or a new one (one of its own when larger).
  uint8_t* alloc(size_t bytes) {
    if (cur && cur->cap - cur->used >= bytes) {
      uint8_t* p = cur->data + cur->used;
      cur->used += bytes;
      return p;
    }
    if (cur) publishLocked();
    const size_t cap = std::max(bytes, kChunkBytes);
    {
      std::unique_lock<std::mutex> lk(qmu);
      const uint64_t q = queued.load(std::memory_order_relaxed);
      if (q && q + cap > kQueueBudget) {
        queueWaits.fetch_add(1, std::memory_order_relaxed);
        freedcv.wait(lk, [&] {
          const uint64_t n = queued.load(std::memory_order_relaxed);
          return !n || n + cap <= kQueueBudget;
        });
      }
      if (bytes <= kChunkBytes && !spare.empty()) {
        cur = spare.back();
        spare.pop_back();
      }
    }
    if (!cur) {
      cur = new CmdChunk(cap);
      if (!cur->data) {
        // Out of address space: let the worker run (and free) everything queued, drop the spares, and try once more.
        chunkAllocFailures.fetch_add(1, std::memory_order_relaxed);
        delete cur;
        cur = nullptr;
        {
          std::unique_lock<std::mutex> lk(qmu);
          freedcv.wait(lk, [&] { return queued.load(std::memory_order_relaxed) == 0; });
          for (CmdChunk* c : spare) delete c;
          spare.clear();
        }
        cur = new CmdChunk(cap);
        if (!cur->data) {
          delete cur;
          cur = nullptr;
          throw std::bad_alloc();
        }
      }
    }
    {
      std::lock_guard<std::mutex> lk(qmu);
      const uint64_t n = queued.fetch_add(cur->cap, std::memory_order_relaxed) + cur->cap;
      if (n > queuedPeak.load(std::memory_order_relaxed)) queuedPeak.store(n, std::memory_order_relaxed);
    }
    cur->used = bytes;
    return cur->data;
  }
  void publishLocked() {
    unpublished = 0;
    if (!cur || !cur->used) return;
    {
      std::lock_guard<std::mutex> lk(qmu);
      ready.push_back(cur);
    }
    cur = nullptr;
    qcv.notify_one();
  }
  // Queues f(payload) with `extra` payload bytes written by fill(payload) (pmu held).
  template <class Fill, class F>
  void postLocked(size_t extra, Fill&& fill, F&& f) {
    using FT = std::decay_t<F>;
    const size_t body = cmdAlign(sizeof(CmdHeader)), pay = cmdAlign(body + sizeof(FT)), total = cmdAlign(pay + extra);
    uint8_t* p = alloc(total);
    CmdHeader* h = new (p) CmdHeader{ &thunk<FT>, uint32_t(total), uint32_t(pay) };
    (void)h;
    fill(p + pay);
    new (p + body) FT(std::forward<F>(f));
    if (++unpublished >= kPublishEvery) publishLocked();
  }
  template <class FT>
  static void thunk(uint8_t* p) {
    const CmdHeader* h = reinterpret_cast<const CmdHeader*>(p);
    FT* f = reinterpret_cast<FT*>(p + cmdAlign(sizeof(CmdHeader)));
    (*f)(static_cast<const uint8_t*>(p + h->payload));
    f->~FT();
  }
  template <class F>
  void post(F&& f) {
    std::lock_guard<std::mutex> lk(pmu);
    postLocked(0, [](uint8_t*) {}, [g = std::forward<F>(f)](const uint8_t*) mutable { g(); });
  }
  template <class Fill, class F>
  void postWith(size_t extra, Fill&& fill, F&& f) {
    std::lock_guard<std::mutex> lk(pmu);
    postLocked(extra, std::forward<Fill>(fill), std::forward<F>(f));
  }
  // Runs f on the worker after everything queued before it, and returns its result.
  template <class F>
  auto sync(F&& f) -> decltype(f()) {
    using R = decltype(f());
    uint64_t id;
    if constexpr (std::is_void_v<R>) {
      {
        std::lock_guard<std::mutex> lk(pmu);
        id = ++syncIssued;
        postLocked(0, [](uint8_t*) {}, [this, &f, id](const uint8_t*) {
          f();
          finishSync(id);
        });
        publishLocked();
      }
      waitSync(id);
    } else {
      R result{};
      {
        std::lock_guard<std::mutex> lk(pmu);
        id = ++syncIssued;
        postLocked(0, [](uint8_t*) {}, [this, &f, &result, id](const uint8_t*) {
          result = f();
          finishSync(id);
        });
        publishLocked();
      }
      waitSync(id);
      return result;
    }
  }
  void finishSync(uint64_t id) {
    publishStats();
    {
      std::lock_guard<std::mutex> lk(dmu);
      syncDone = id;
    }
    dcv.notify_all();
  }
  void waitSync(uint64_t id) {
    const uint64_t t0 = nowUs();
    {
      std::unique_lock<std::mutex> lk(dmu);
      dcv.wait(lk, [&] { return syncDone >= id; });
    }
    syncWaitUs.fetch_add(nowUs() - t0, std::memory_order_relaxed);
    syncs.fetch_add(1, std::memory_order_relaxed);
  }

  void run() {
    for (;;) {
      CmdChunk* c = nullptr;
      {
        std::unique_lock<std::mutex> lk(qmu);
        // With queries waiting for the GPU, idle waits are short: their results are published even when the caller
        // queues nothing more (it may be polling one of them).
        while (!stopping && ready.empty()) {
          if (pending.empty()) {
            qcv.wait(lk);
          } else if (!qcv.wait_for(lk, std::chrono::milliseconds(1), [&] { return stopping || !ready.empty(); })) {
            lk.unlock();
            pollQueries();
            lk.lock();
          }
        }
        if (ready.empty()) return;  // stopping, and everything queued has run
        c = ready.front();
        ready.pop_front();
      }
      const uint64_t t0 = nowUs();
      for (size_t off = 0; off < c->used;) {
        CmdHeader* h = reinterpret_cast<CmdHeader*>(c->data + off);
        const uint32_t size = h->size;
        h->run(c->data + off);
        off += size;
      }
      pollQueries();
      busyUs.fetch_add(nowUs() - t0, std::memory_order_relaxed);
      c->used = 0;
      {
        std::lock_guard<std::mutex> lk(qmu);
        queued.fetch_sub(c->cap, std::memory_order_relaxed);
        if (c->cap == kChunkBytes && spare.size() < 32) spare.push_back(c);
        else delete c;
      }
      freedcv.notify_all();
    }
  }
  void start() { worker = std::thread([this] { run(); }); }
  void stop() {
    {
      std::lock_guard<std::mutex> lk(pmu);
      publishLocked();
    }
    {
      std::lock_guard<std::mutex> lk(qmu);
      stopping = true;
    }
    qcv.notify_one();
    if (worker.joinable()) worker.join();
    for (CmdChunk* c : spare) delete c;
    spare.clear();
    delete cur;
    cur = nullptr;
  }

  // --- frames ----------------------------------------------------------------------------------------------------
  std::atomic<uint64_t> frames{ 0 };
  void present(RtKey onscreen) {
    uint64_t n;
    {
      std::lock_guard<std::mutex> lk(pmu);
      n = ++presentsIssued;
      postLocked(0, [](uint8_t*) {}, [this, onscreen, n](const uint8_t*) {
        {
          std::lock_guard<std::mutex> lock(m.mutex);
          FpuGuard fpu;
          m.doPresent(onscreen);
        }
        publishStats();
        {
          std::lock_guard<std::mutex> lk(dmu);
          presentsDone = n;
        }
        dcv.notify_all();
      });
      publishLocked();
    }
    frames.fetch_add(1, std::memory_order_relaxed);
    const uint64_t t0 = nowUs();
    {
      std::unique_lock<std::mutex> lk(dmu);
      dcv.wait(lk, [&] { return presentsDone + 1 >= n; });
    }
    presentWaitUs.fetch_add(nowUs() - t0, std::memory_order_relaxed);
  }

  // --- stats published by the worker -----------------------------------------------------------------------------
  mutable std::mutex pubmu;
  BackendStats snapshot;  // pubmu
  void publishStats() {
    BackendStats s;
    {
      std::lock_guard<std::mutex> lock(m.mutex);
      s = m.collectStats();
    }
    std::lock_guard<std::mutex> lk(pubmu);
    snapshot = s;
  }

  // --- the caller's view of objects ------------------------------------------------------------------------------
  std::unordered_map<ShaderKey, const PackedShader*> shaders;  // pmu
  std::unordered_set<TexKey> textures;                         // pmu: created by createTexture and not destroyed
  const PackedShader* shaderOf(ShaderKey k) {
    auto it = shaders.find(k);
    return it == shaders.end() ? nullptr : it->second;
  }

  // --- occlusion queries -----------------------------------------------------------------------------------------
  uint32_t nextQueryId = 1;                        // pmu
  std::unordered_map<uint32_t, uint64_t> queryEnds;  // pmu: query -> number of its latest end (0: never ended)
  struct Pending { uint32_t id; uint64_t seq; };
  std::vector<Pending> pending;                    // worker
  struct Published { uint64_t seq; uint32_t samples; };
  std::unordered_map<uint32_t, Published> results;  // pubmu
  void pollQueries() {
    if (pending.empty()) return;
    std::lock_guard<std::mutex> lock(m.mutex);
    for (size_t i = 0; i < pending.size();) {
      uint32_t samples = 0;
      if (m.doQueryResult(pending[i].id, samples)) {
        {
          std::lock_guard<std::mutex> lk(pubmu);
          results[pending[i].id] = { pending[i].seq, samples };
        }
        pending[i] = pending.back();
        pending.pop_back();
      } else {
        i++;
      }
    }
  }

  // --- sun gate --------------------------------------------------------------------------------------------------
  bool sunOn = false;                // pmu
  bool (*gate)(void*) = nullptr;     // pmu
  void* gateCtx = nullptr;           // pmu

  // --- draws -----------------------------------------------------------------------------------------------------
  // The caller's copies of what was last sent, and the worker's arrays the commands apply to.
  uint32_t rs[256] = {}, ss[16][14] = {};
  bool rsValid = false, ssValid = false;
  uint64_t rsVersion = 0;
  float vsc[256][4] = {}, psc[224][4] = {};
  struct Sent {
    const PackedShader* shader = nullptr;
    uint64_t version = 0;
    bool valid = false;
  } vsSent, psSent;
  uint32_t wrs[256] = {}, wss[16][14] = {};
  float wvsc[256][4] = {}, wpsc[224][4] = {};

  struct Delta { uint16_t first, count; };
  // The runs of `regs` to send for this draw's shader (pmu held); updates the caller's copy.
  uint32_t planConstants(const float* regs, uint32_t count, const PackedShader* sh, uint64_t version, Sent& sent,
                         float (*mirror)[4], Delta* out) {
    if (!regs) {
      sent.valid = false;
      return 0;
    }
    if (!sh) return 0;  // not drawn (no shader)
    if (version != 0 && sent.valid && sent.shader == sh && sent.version == version) return 0;
    uint32_t n = 0;
    auto consider = [&](uint32_t first, uint32_t cnt) {
      if (first >= count) return;
      cnt = std::min(cnt, count - first);
      if (memcmp(regs + first * 4, mirror[first], cnt * 16) == 0) return;
      memcpy(mirror[first], regs + first * 4, cnt * 16);
      out[n++] = { uint16_t(first), uint16_t(cnt) };
    };
    if (sh->constAll) consider(0, count);
    else
      for (const ConstRun& r : sh->constRuns) consider(r.first, r.count);
    sent = { sh, version, true };
    return n;
  }

  // A draw is sent as what changed since the previous one (the caller's `prev`, the worker's `wcur`): a fixed head, then
  // in order the shader / declaration / index keys, the texture slots and streams that changed, the viewport, scissor
  // and clip planes when they changed, the render and sampler states, and the constant runs (planConstants).
  DrawSetup prev{}, wcur{};
  bool prevValid = false;
  enum : uint32_t {
    kIndexed = 1u << 0, kRsNull = 1u << 1, kSsNull = 1u << 2, kVsNull = 1u << 3, kPsNull = 1u << 4,
    kSendRs = 1u << 5, kSendSs = 1u << 6, kSendViewport = 1u << 7, kSendScissor = 1u << 8, kSendClip = 1u << 9,
    kSendKeys = 1u << 10,
  };
  struct DrawHead {
    uint32_t pt, prims, start;
    int32_t baseVertex;
    uint64_t vsVersion, psVersion, rsVersion, clipVersion;
    uint32_t flags;
    uint16_t texMask;     // texture slots that follow
    uint8_t streamMask;   // streams that follow
    int8_t gate;          // SunLighting::gate's answer, -1 none
    uint8_t vsDeltas, psDeltas;
  };
  struct Keys { ShaderKey vs, ps; DeclKey decl; BufKey indices; };
  // shaderOf for the last keys asked (most draws repeat their shaders); createShader / destroyShader clear it.
  ShaderKey cachedKey[2] = {};
  const PackedShader* cachedShader[2] = {};
  const PackedShader* shaderCached(uint32_t stage, ShaderKey k) {
    if (k && cachedKey[stage] == k) return cachedShader[stage];
    cachedKey[stage] = k;
    return cachedShader[stage] = shaderOf(k);
  }
  void forgetShaders() { cachedKey[0] = cachedKey[1] = 0; }

  void draw(const DrawSetup& in, uint32_t pt, uint32_t prims, bool indexed, int32_t baseVertex, uint32_t start) {
    std::lock_guard<std::mutex> lk(pmu);
    DrawHead h{ pt, prims, start, baseVertex, in.vsConstantsVersion, in.psConstantsVersion, in.renderStatesVersion,
                in.clipPlanesVersion, 0u, 0, 0, -1, 0, 0 };
    if (indexed) h.flags |= kIndexed;
    if (!in.renderStates) h.flags |= kRsNull;
    if (!in.samplerStates) h.flags |= kSsNull;
    if (!in.vsConstants) h.flags |= kVsNull;
    if (!in.psConstants) h.flags |= kPsNull;
    size_t extra = sizeof(DrawHead);
    if (!prevValid || in.vs != prev.vs || in.ps != prev.ps || in.decl != prev.decl || in.indices != prev.indices) {
      h.flags |= kSendKeys;
      extra += sizeof(Keys);
    }
    for (uint32_t i = 0; i < 16; i++)
      if (!prevValid || in.textures[i].kind != prev.textures[i].kind || in.textures[i].key != prev.textures[i].key) {
        h.texMask |= uint16_t(1u << i);
        extra += sizeof(DrawSetup::Texture);
      }
    for (uint32_t i = 0; i < 8; i++)
      if (!prevValid || memcmp(&in.streams[i], &prev.streams[i], sizeof(DrawSetup::Stream)) != 0) {
        h.streamMask |= uint8_t(1u << i);
        extra += sizeof(DrawSetup::Stream);
      }
    if (!prevValid || memcmp(&in.viewport, &prev.viewport, sizeof(in.viewport)) != 0) {
      h.flags |= kSendViewport;
      extra += sizeof(in.viewport);
    }
    if (!prevValid || memcmp(in.scissor, prev.scissor, sizeof(in.scissor)) != 0) {
      h.flags |= kSendScissor;
      extra += sizeof(in.scissor);
    }
    if (!prevValid || in.clipPlanesVersion == 0 || in.clipPlanesVersion != prev.clipPlanesVersion) {
      if (!prevValid || memcmp(in.clipPlanes, prev.clipPlanes, sizeof(in.clipPlanes)) != 0) {
        h.flags |= kSendClip;
        extra += sizeof(in.clipPlanes);
      }
    }
    if (in.renderStates && !(in.renderStatesVersion != 0 && rsValid && in.renderStatesVersion == rsVersion)) {
      if (!rsValid || memcmp(rs, in.renderStates, sizeof(rs)) != 0) {
        memcpy(rs, in.renderStates, sizeof(rs));
        h.flags |= kSendRs;
        extra += sizeof(rs);
      }
      rsValid = true;
      rsVersion = in.renderStatesVersion;
    }
    if (in.samplerStates) {
      if (!ssValid || memcmp(ss, in.samplerStates, sizeof(ss)) != 0) {
        memcpy(ss, in.samplerStates, sizeof(ss));
        h.flags |= kSendSs;
        extra += sizeof(ss);
      }
      ssValid = true;
    }
    Delta vd[10], pd[10];
    h.vsDeltas = uint8_t(planConstants(in.vsConstants, 256, shaderCached(0, in.vs), in.vsConstantsVersion, vsSent, vsc, vd));
    h.psDeltas = uint8_t(planConstants(in.psConstants, 224, shaderCached(1, in.ps), in.psConstantsVersion, psSent, psc, pd));
    for (uint32_t i = 0; i < h.vsDeltas; i++) extra += sizeof(Delta) + vd[i].count * 16u;
    for (uint32_t i = 0; i < h.psDeltas; i++) extra += sizeof(Delta) + pd[i].count * 16u;
    h.gate = sunOn && gate ? int8_t(gate(gateCtx) ? 1 : 0) : int8_t(-1);
    postLocked(
        extra,
        [&](uint8_t* p) {
          auto put = [&](const void* src, size_t n) { memcpy(p, src, n); p += n; };
          put(&h, sizeof(h));
          if (h.flags & kSendKeys) {
            const Keys k{ in.vs, in.ps, in.decl, in.indices };
            put(&k, sizeof(k));
          }
          for (uint32_t i = 0; i < 16; i++)
            if (h.texMask >> i & 1u) put(&in.textures[i], sizeof(DrawSetup::Texture));
          for (uint32_t i = 0; i < 8; i++)
            if (h.streamMask >> i & 1u) put(&in.streams[i], sizeof(DrawSetup::Stream));
          if (h.flags & kSendViewport) put(&in.viewport, sizeof(in.viewport));
          if (h.flags & kSendScissor) put(in.scissor, sizeof(in.scissor));
          if (h.flags & kSendClip) put(in.clipPlanes, sizeof(in.clipPlanes));
          if (h.flags & kSendRs) put(rs, sizeof(rs));
          if (h.flags & kSendSs) put(ss, sizeof(ss));
          for (uint32_t i = 0; i < h.vsDeltas; i++) { put(&vd[i], sizeof(Delta)); put(vsc[vd[i].first], vd[i].count * 16u); }
          for (uint32_t i = 0; i < h.psDeltas; i++) { put(&pd[i], sizeof(Delta)); put(psc[pd[i].first], pd[i].count * 16u); }
        },
        [this](const uint8_t* p) { runDraw(p); });
    prev = in;  // only its value fields are compared
    prevValid = true;
  }
  void runDraw(const uint8_t* p) {
    auto take = [&](void* dst, size_t n) { memcpy(dst, p, n); p += n; };
    DrawHead h;
    take(&h, sizeof(h));
    DrawSetup& s = wcur;
    if (h.flags & kSendKeys) {
      Keys k;
      take(&k, sizeof(k));
      s.vs = k.vs;
      s.ps = k.ps;
      s.decl = k.decl;
      s.indices = k.indices;
    }
    for (uint32_t i = 0; i < 16; i++)
      if (h.texMask >> i & 1u) take(&s.textures[i], sizeof(DrawSetup::Texture));
    for (uint32_t i = 0; i < 8; i++)
      if (h.streamMask >> i & 1u) take(&s.streams[i], sizeof(DrawSetup::Stream));
    if (h.flags & kSendViewport) take(&s.viewport, sizeof(s.viewport));
    if (h.flags & kSendScissor) take(s.scissor, sizeof(s.scissor));
    if (h.flags & kSendClip) take(s.clipPlanes, sizeof(s.clipPlanes));
    if (h.flags & kSendRs) take(wrs, sizeof(wrs));
    if (h.flags & kSendSs) take(wss, sizeof(wss));
    for (uint32_t i = 0; i < h.vsDeltas; i++) {
      Delta d;
      take(&d, sizeof(d));
      take(wvsc[d.first], d.count * 16u);
    }
    for (uint32_t i = 0; i < h.psDeltas; i++) {
      Delta d;
      take(&d, sizeof(d));
      take(wpsc[d.first], d.count * 16u);
    }
    s.vsConstantsVersion = h.vsVersion;
    s.psConstantsVersion = h.psVersion;
    s.renderStatesVersion = h.rsVersion;
    s.clipPlanesVersion = h.clipVersion;
    s.renderStates = (h.flags & kRsNull) ? nullptr : wrs;
    s.samplerStates = (h.flags & kSsNull) ? nullptr : wss;
    s.vsConstants = (h.flags & kVsNull) ? nullptr : &wvsc[0][0];
    s.psConstants = (h.flags & kPsNull) ? nullptr : &wpsc[0][0];
    std::lock_guard<std::mutex> lock(m.mutex);
    FpuGuard fpu;
    m.sunGateOverride = h.gate;
    if (m.initialized) m.doDraw(s, h.pt, h.prims, (h.flags & kIndexed) != 0, h.baseVertex, h.start);
    m.sunGateOverride = -1;
  }
};
std::unique_ptr<Recorder> rec;
std::atomic<bool> readyFlag{ false };
int8_t sunGateOverride = -1;  // the record thread's answer of SunLighting::gate for the draw being recorded; -1: ask it

// Without the record thread f runs now; with it, f is queued (post) or run after the queue drains (call).
template <class F>
void post(F&& f) {
  if (rec) rec->post(std::forward<F>(f));
  else f();
}
template <class F>
auto call(F&& f) -> decltype(f()) {
  if (rec) return rec->sync(f);
  return f();
}

BackendStats collectStats() const {
  BackendStats s = st;
  s.liveBuffers = uint32_t(buffers.size());
  s.live = uint32_t(rts.size());
  s.liveRealized = 0;
  for (auto& [k, rt] : rts) s.liveRealized += rt.realized ? 1 : 0;
  s.liveTextures = uint32_t(textures.size());
  if (rec) {
    s.recordBusyUs = rec->busyUs.load(std::memory_order_relaxed);
    s.recordPresentWaitUs = rec->presentWaitUs.load(std::memory_order_relaxed);
    s.recordSyncWaitUs = rec->syncWaitUs.load(std::memory_order_relaxed);
    s.recordSyncs = rec->syncs.load(std::memory_order_relaxed);
    s.textureQueuePeakBytes = rec->texQueuedPeak.load(std::memory_order_relaxed);
    s.textureQueueWaits = rec->texWaits.load(std::memory_order_relaxed);
  }
  return s;
}
