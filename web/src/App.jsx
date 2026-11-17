import { useCallback, useEffect, useRef, useState } from "react";

const API = "/api";

async function requestJson(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function formatBytes(bytes) {
  if (!bytes) return "0 MB";
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds) {
  if (!seconds) return "0:00";
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function App() {
  const [page, setPage] = useState("overview");
  const [health, setHealth] = useState(null);
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState("");
  const [upload, setUpload] = useState(null);
  const [confidence, setConfidence] = useState(0.22);
  const [maxFrames, setMaxFrames] = useState(300);
  const [enableDepth, setEnableDepth] = useState(false);
  const [run, setRun] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [detections, setDetections] = useState([]);
  const [cameraImage, setCameraImage] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const videoRef = useRef(null);
  const cameraBusyRef = useRef(false);
  const streamRef = useRef(null);
  const uploadInputRef = useRef(null);
  const lastFrameAt = useRef(0);

  const loadDashboard = useCallback(async () => {
    try {
      const [healthData, videoData] = await Promise.all([
        requestJson("/health"),
        requestJson("/videos"),
      ]);
      setHealth(healthData);
      setVideos(videoData.videos);
      setError("");
    } catch (requestError) {
      setHealth(null);
      setError(requestError.message);
    }
  }, []);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  useEffect(() => {
    if (!running || !run?.id) return undefined;
    const started = Date.now();
    const timer = window.setInterval(async () => {
      setElapsed(Math.floor((Date.now() - started) / 1000));
      try {
        const status = await requestJson(`/runs/${run.id}`);
        setRun(status);
        if (status.status === "completed" || status.status === "failed") {
          setRunning(false);
          if (status.status === "failed") setError(status.error || "Video processing failed.");
        }
      } catch (requestError) {
        setRunning(false);
        setError(requestError.message);
      }
    }, 1400);
    return () => window.clearInterval(timer);
  }, [run?.id, running]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    cameraBusyRef.current = false;
    setCameraActive(false);
  }, []);

  useEffect(() => () => {
    if (streamRef.current) streamRef.current.getTracks().forEach((track) => track.stop());
  }, []);

  useEffect(() => {
    if (!cameraActive) return undefined;
    let cancelled = false;
    let timer;
    const captureFrame = async () => {
      if (cancelled || cameraBusyRef.current || !videoRef.current || videoRef.current.readyState < 2) {
        timer = window.setTimeout(captureFrame, 200);
        return;
      }
      cameraBusyRef.current = true;
      const video = videoRef.current;
      const canvas = document.createElement("canvas");
      const scale = Math.min(1, 960 / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      try {
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
        if (!blob) throw new Error("Could not capture a camera frame.");
        const response = await fetch(`${API}/live/detect?confidence=${confidence}`, {
          method: "POST",
          headers: { "Content-Type": "image/jpeg" },
          body: blob,
        });
        if (!response.ok) {
          const result = await response.json().catch(() => ({}));
          throw new Error(result.detail || "Live detection request failed.");
        }
        const result = await response.json();
        if (!cancelled) {
          setCameraImage(`data:image/jpeg;base64,${result.image}`);
          setDetections(result.detections);
          lastFrameAt.current = Date.now();
          setCameraError("");
        }
      } catch (requestError) {
        if (!cancelled) setCameraError(requestError.message);
      } finally {
        cameraBusyRef.current = false;
        timer = window.setTimeout(captureFrame, 350);
      }
    };
    captureFrame();
    return () => {
      cancelled = true;
      cameraBusyRef.current = false;
      window.clearTimeout(timer);
    };
  }, [cameraActive, confidence]);

  async function startCamera() {
    setCameraError("");
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access requires HTTPS or localhost in a supported browser.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraImage("");
      setCameraActive(true);
    } catch (cameraRequestError) {
      setCameraError(cameraRequestError.message || "Could not access the camera.");
      stopCamera();
    }
  }

  async function startRun(event) {
    event.preventDefault();
    setError("");
    setRun(null);
    setElapsed(0);
    const body = new FormData();
    if (upload) body.append("upload", upload);
    else if (selectedVideo) body.append("video_name", selectedVideo);
    else {
      setError("Select a project video or upload a video first.");
      return;
    }
    body.append("confidence", String(confidence));
    body.append("enable_depth", String(enableDepth));
    body.append("max_frames", String(maxFrames));
    setRunning(true);
    try {
      const created = await requestJson("/runs", { method: "POST", body });
      setRun(created);
    } catch (requestError) {
      setRunning(false);
      setError(requestError.message);
    }
  }

  const selectedProjectVideo = videos.find((video) => video.name === selectedVideo);
  const previewVideo = run?.status === "completed" ? run.video_url : selectedProjectVideo?.url;
  const appReady = health?.status === "ok";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">A</div>
          <div><strong>ASTRA</strong><span>ADAS PLATFORM</span></div>
        </div>
        <div className="nav-label">WORKSPACE</div>
        <button className={`nav-item ${page === "overview" ? "selected" : ""}`} onClick={() => setPage("overview")}>
          <span className="nav-icon">◫</span> Overview
        </button>
        <button className={`nav-item ${page === "camera" ? "selected" : ""}`} onClick={() => setPage("camera")}>
          <span className="nav-icon">◉</span> Live camera
        </button>
        <div className="sidebar-bottom">
          <div className="system-card">
            <span className={`status-dot ${appReady ? "online" : ""}`} />
            <div><strong>{appReady ? "System online" : "API disconnected"}</strong><small>{appReady ? "Inference service ready" : "Start the backend service"}</small></div>
          </div>
          <div className="sidebar-foot">ASTRA ADAS <span>RESEARCH BUILD</span></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">Astra ADAS <span>/</span> {page === "overview" ? "Overview" : "Live camera"}</div>
          <div className="topbar-right">
            <span className={`connection-pill ${appReady ? "connected" : ""}`}><i />{appReady ? "Backend connected" : "Connecting"}</span>
            <div className="avatar">LA</div>
          </div>
        </header>

        <div className="content">
          {error && <div className="alert error-alert"><span>!</span>{error}<button onClick={() => setError("")}>×</button></div>}

          {page === "overview" ? (
            <>
              <section className="page-heading">
                <div>
                  <div className="eyebrow"><span className="eyebrow-line" /> ROAD SCENE INTELLIGENCE</div>
                  <h1>Perception dashboard</h1>
                  <p>Analyze road footage, inspect object detections, and review ADAS pipeline outputs.</p>
                </div>
                <button className="secondary-button" onClick={loadDashboard}>↻ <span>Refresh data</span></button>
              </section>

              <section className="metric-grid">
                <MetricCard label="DETECTION CLASSES" value="15" detail="IDD road-user categories" icon="◈" accent="blue" />
                <MetricCard label="TEST IMAGES EVALUATED" value="4,197" detail="Full held-out IDD-D split" icon="▧" accent="purple" />
                <MetricCard label="DETECTOR mAP@50" value="37.1%" detail="IDD model · test split" icon="⌖" accent="green" />
                <MetricCard label="MODEL THROUGHPUT" value="34.5" suffix="FPS" detail="CPU detector benchmark" icon="ϟ" accent="orange" />
              </section>

              <section className="work-grid">
                <div className="panel analysis-panel">
                  <div className="panel-heading">
                    <div><span className="panel-kicker">VIDEO ANALYSIS</span><h2>Start a new analysis</h2></div>
                    <span className="pipeline-badge"><i /> PIPELINE READY</span>
                  </div>
                  <form onSubmit={startRun}>
                    <label className="field-label">VIDEO SOURCE</label>
                    <div className="source-tabs">
                      <button type="button" className={`source-tab ${!upload ? "active" : ""}`} onClick={() => { setUpload(null); if (uploadInputRef.current) uploadInputRef.current.value = ""; }}>Project library</button>
                      <button type="button" className={`source-tab ${upload ? "active" : ""}`} onClick={() => uploadInputRef.current?.click()}>Upload video</button>
                    </div>
                    <input ref={uploadInputRef} className="hidden-input" type="file" accept=".mp4,.avi,.mov,video/*" onChange={(event) => { setUpload(event.target.files?.[0] || null); setSelectedVideo(""); }} />
                    {upload ? (
                      <div className="selected-file"><span className="file-icon">▶</span><div><strong>{upload.name}</strong><small>{formatBytes(upload.size)} · ready to analyze</small></div><button type="button" onClick={() => { setUpload(null); if (uploadInputRef.current) uploadInputRef.current.value = ""; }}>×</button></div>
                    ) : (
                      <select className="select-input" value={selectedVideo} onChange={(event) => setSelectedVideo(event.target.value)}>
                        <option value="">Choose a video from outputs/videos…</option>
                        {videos.map((video) => <option key={video.name} value={video.name}>{video.name} · {formatBytes(video.size_bytes)}</option>)}
                      </select>
                    )}
                    {!videos.length && !upload && <div className="field-hint">No project videos found. Upload an MP4, AVI, or MOV to get started.</div>}

                    <div className="settings-row">
                      <div className="setting-block">
                        <div className="setting-title"><label className="field-label" htmlFor="confidence">DETECTION CONFIDENCE</label><strong>{confidence.toFixed(2)}</strong></div>
                        <input id="confidence" className="range-input" type="range" min="0.15" max="0.80" step="0.05" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} />
                        <div className="range-labels"><span>More detections</span><span>Higher certainty</span></div>
                      </div>
                      <div className="setting-block frames-setting">
                        <label className="field-label" htmlFor="max-frames">FRAME LIMIT</label>
                        <select id="max-frames" className="select-input compact" value={maxFrames} onChange={(event) => setMaxFrames(Number(event.target.value))}>
                          <option value={300}>300 frames</option><option value={1000}>1,000 frames</option><option value={3000}>3,000 frames</option><option value={0}>All frames</option>
                        </select>
                      </div>
                    </div>
                    <label className="toggle-row">
                      <span className="toggle-copy"><strong>MiDaS depth estimation</strong><small>Optional relative-depth map · adds processing time</small></span>
                      <input type="checkbox" checked={enableDepth} onChange={(event) => setEnableDepth(event.target.checked)} />
                      <span className="toggle-switch" />
                    </label>
                    <button className="primary-button run-button" type="submit" disabled={!appReady || running}>
                      {running ? <><span className="spinner" /> Processing video…</> : <>▶ <span>Run ADAS analysis</span><b>→</b></>}
                    </button>
                    <div className="safety-note">Prototype output only. Distance and TTC values are uncalibrated estimates.</div>
                  </form>
                </div>

                <div className="panel preview-panel">
                  <div className="panel-heading">
                    <div><span className="panel-kicker">FRAME PREVIEW</span><h2>{run?.status === "completed" ? "Annotated result" : "Selected source"}</h2></div>
                    {run && <span className={`run-state ${run.status}`}>{run.status}</span>}
                  </div>
                  <div className="video-stage">
                    {previewVideo ? (
                      <video key={previewVideo} controls playsInline src={previewVideo} />
                    ) : (
                      <div className="empty-preview">
                        <div className="road-art"><span /><i /><b /></div>
                        <strong>Your analysis preview</strong>
                        <small>Select a project video or upload a road clip</small>
                      </div>
                    )}
                    {running && <div className="processing-overlay"><span className="spinner large" /><strong>Analyzing road footage</strong><small>Elapsed {formatDuration(elapsed)} · processing frame stream</small></div>}
                  </div>
                  {run?.status === "completed" ? (
                    <div className="download-row">
                      <span>{run.record_count ?? 0} object records</span>
                      <a className="download-button" href={run.video_url} download>↓ Annotated video</a>
                      <a className="download-button outline" href={run.metrics_url} download>↓ Metrics CSV</a>
                    </div>
                  ) : (
                    <div className="preview-meta">
                      <span><i className="meta-dot" /> {selectedProjectVideo ? "Source ready" : "Waiting for video"}</span>
                      <span>MP4 · AVI · MOV</span>
                    </div>
                  )}
                  {run?.status === "failed" && <div className="run-error">{run.error}</div>}
                </div>
              </section>

              <section className="panel recent-panel">
                <div className="panel-heading">
                  <div><span className="panel-kicker">PROCESSING PIPELINE</span><h2>What happens to each frame</h2></div>
                  <span className="light-tag">RESEARCH PROTOTYPE</span>
                </div>
                <div className="pipeline-steps">
                  {["YOLO detection", "ByteTrack IDs", "Lane + signals", "Motion + TTC", "Risk + decision"].map((item, index) => (
                    <div className="pipeline-step" key={item}><span>{String(index + 1).padStart(2, "0")}</span><strong>{item}</strong>{index < 4 && <i>→</i>}</div>
                  ))}
                </div>
              </section>
            </>
          ) : (
            <section className="camera-page">
              <div className="page-heading">
                <div><div className="eyebrow"><span className="eyebrow-line" /> BROWSER WEBCAM · YOLO OBJECT DETECTION</div><h1>Live camera</h1><p>Detect configured IDD road users using your browser camera.</p></div>
                {cameraActive ? <button className="stop-button" onClick={stopCamera}>■ <span>Stop camera</span></button> : <button className="primary-button camera-start" disabled={!appReady} onClick={startCamera}>◉ <span>Start camera</span></button>}
              </div>
              {cameraError && <div className="alert error-alert"><span>!</span>{cameraError}</div>}
              <div className="camera-layout">
                <div className="panel camera-panel">
                  <div className="panel-heading"><div><span className="panel-kicker">LIVE FEED</span><h2>Camera view</h2></div>{cameraActive && <span className="live-badge"><i /> LIVE</span>}</div>
                  <div className={`camera-stage ${cameraActive ? "active" : ""}`}>
                    <video ref={videoRef} className="camera-source" autoPlay muted playsInline />
                    {cameraActive && cameraImage ? <img src={cameraImage} className="camera-annotated" alt="Annotated live camera feed" /> : null}
                    {!cameraActive && <div className="camera-placeholder"><div className="camera-glyph">◉</div><strong>Camera is off</strong><small>Start the camera and allow browser access</small></div>}
                    {cameraActive && !cameraImage && <div className="camera-placeholder compact-placeholder"><span className="spinner" /><small>Loading first frame…</small></div>}
                    <div className="camera-hud"><span>{cameraActive ? "LOCAL CAMERA" : "NO SIGNAL"}</span><span>{cameraActive && lastFrameAt.current ? "YOLO · IDD 15 CLASSES" : ""}</span></div>
                  </div>
                  <div className="camera-foot"><span>Camera frames are sent to the local Astra ADAS API for inference.</span><span>Frame updates are rate-limited for CPU workloads.</span></div>
                </div>
                <div className="panel detections-panel">
                  <div className="panel-heading"><div><span className="panel-kicker">CURRENT FRAME</span><h2>Detected objects</h2></div><span className="count-badge">{detections.length}</span></div>
                  <div className="confidence-setting">
                    <div><span>Confidence threshold</span><strong>{confidence.toFixed(2)}</strong></div>
                    <input className="range-input" type="range" min="0.15" max="0.80" step="0.05" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} />
                  </div>
                  <div className="detection-list">
                    {!cameraActive && <div className="detection-empty">Start the camera to see detections here.</div>}
                    {cameraActive && !detections.length && cameraImage && <div className="detection-empty">No configured road-user classes detected.</div>}
                    {detections.map((detection, index) => (
                      <div className="detection-item" key={`${detection.track_id}-${detection.class_name}-${index}`}>
                        <span className="detection-symbol">◈</span>
                        <div className="detection-name"><strong>{detection.class_name}</strong><small>{detection.track_id == null ? "Untracked candidate" : `Track ID ${detection.track_id}`}</small></div>
                        <span className="detection-confidence">{(detection.confidence * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                  <div className="disclaimer-box"><strong>Prototype notice</strong><p>This live view displays object detections only. It does not run lane, distance, TTC, risk, or decision analysis.</p></div>
                </div>
              </div>
            </section>
          )}
        </div>
        <footer className="footer"><span>ASTRA ADAS · RESEARCH PROTOTYPE</span><span>Not a certified vehicle safety system</span></footer>
      </main>
    </div>
  );
}

function MetricCard({ label, value, suffix, detail, icon, accent }) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${accent}`}>{icon}</div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}{suffix && <small>{suffix}</small>}</div>
      <div className="metric-detail">{detail}</div>
    </div>
  );
}

export default App;
