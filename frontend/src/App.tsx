import { useEffect, useRef, useState } from "react";

import { featureDensity } from "./density";

type FrameInfo = {
  frame_id: number;
  width: number;
  height: number;
  fps: number;
};

type VisionInfo = {
  frame_id: number;
  feature_count: number;
  keypoints: [number, number, number, number, number, number][];
  detector_fps: number;
  detection_time_ms: number;
};

const backend =
  import.meta.env.VITE_BACKEND_URL ??
  `ws://${window.location.hostname}:8000/ws`;

export default function App() {
  const [status, setStatus] = useState("connecting");
  const [message, setMessage] = useState("");
  const [frame, setFrame] = useState<FrameInfo | null>(null);
  const [vision, setVision] = useState<VisionInfo | null>(null);
  const [imageUrl, setImageUrl] = useState("");
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const socket = new WebSocket(backend);
    socket.binaryType = "arraybuffer";
    let pendingFrame: FrameInfo | null = null;
    let currentUrl = "";
    let active = true;

    socket.onmessage = (event) => {
      if (!active) return;
      if (typeof event.data === "string") {
        const data = JSON.parse(event.data);
        if (data.type === "status") {
          setStatus(data.status);
          setMessage(data.message);
        } else if (data.type === "frame") {
          pendingFrame = data;
          setFrame(data);
        } else if (data.type === "vision") {
          setVision({
            frame_id: data.frame_id,
            feature_count: data.feature_count,
            keypoints: data.keypoints,
            detector_fps: data.detector_fps,
            detection_time_ms: data.detection_time_ms,
          });
        }
        return;
      }

      if (pendingFrame) {
        const nextUrl = URL.createObjectURL(
          new Blob([event.data], { type: "image/jpeg" }),
        );
        setImageUrl(nextUrl);
        if (currentUrl) URL.revokeObjectURL(currentUrl);
        currentUrl = nextUrl;
        pendingFrame = null;
      }
    };
    socket.onerror = () => active && setStatus("backend unavailable");
    socket.onclose = () => active && setStatus("backend disconnected");

    return () => {
      active = false;
      socket.close();
      if (currentUrl) URL.revokeObjectURL(currentUrl);
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    context.clearRect(0, 0, frame.width, frame.height);
    if (!vision) return;
    context.strokeStyle = "rgba(75, 255, 167, 0.9)";
    context.lineWidth = 1.5;
    for (const [x, y, size] of vision.keypoints) {
      context.beginPath();
      context.arc(x, y, Math.max(2, Math.min(size / 2, 8)), 0, Math.PI * 2);
      context.stroke();
    }
  }, [frame, vision]);

  const density = vision ? featureDensity(vision.feature_count) : null;
  const value = (current: string | number | undefined) => current ?? "--";

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">Realtime perception</p>
          <h1>Vision Pipeline</h1>
        </div>
        <span className={`connection ${status}`}>{status}</span>
      </header>

      <div className="workspace">
        <section className="preview-panel" aria-label="Live camera with features">
          <div
            className="camera-stack"
            style={{ aspectRatio: frame ? `${frame.width}/${frame.height}` : "16/9" }}
          >
            {imageUrl ? (
              <img src={imageUrl} alt="Live RTSP camera preview" />
            ) : (
              <div className="empty">
                {message || "Waiting for camera frames…"}
              </div>
            )}
            <canvas
              ref={canvasRef}
              width={frame?.width ?? 1}
              height={frame?.height ?? 1}
              aria-hidden="true"
            />
          </div>
          <div className="preview-caption">
            <span>ORB keypoints</span>
            <span>Frame {value(vision?.frame_id)}</span>
          </div>
        </section>

        <aside aria-label="Realtime diagnostics">
          <h2>Diagnostics</h2>

          <section className="diagnostic-group">
            <h3>Streaming</h3>
            <dl>
              <div><dt>Connection Status</dt><dd>{status}</dd></div>
              <div><dt>FPS</dt><dd>{frame?.fps.toFixed(1) ?? "--"}</dd></div>
              <div><dt>Frame ID</dt><dd>{value(frame?.frame_id)}</dd></div>
              <div>
                <dt>Resolution</dt>
                <dd>{frame ? `${frame.width} × ${frame.height}` : "--"}</dd>
              </div>
            </dl>
          </section>

          <section className="diagnostic-group">
            <h3>Vision</h3>
            <dl>
              <div><dt>Feature Count</dt><dd>{value(vision?.feature_count)}</dd></div>
              <div><dt>Detector FPS</dt><dd>{vision?.detector_fps.toFixed(1) ?? "--"}</dd></div>
              <div><dt>Detection Time</dt><dd>{vision ? `${vision.detection_time_ms.toFixed(1)} ms` : "--"}</dd></div>
              <div>
                <dt>Feature Density</dt>
                <dd className={density ? `density ${density.toLowerCase()}` : ""}>
                  {density ?? "--"}
                </dd>
              </div>
            </dl>
          </section>

          <section className="diagnostic-group muted">
            <h3>Tracking</h3>
            <dl>
              <div><dt>Tracked Features</dt><dd>--</dd></div>
              <div><dt>Tracking Confidence</dt><dd>--</dd></div>
            </dl>
          </section>

          <section className="diagnostic-group muted">
            <h3>Mapping</h3>
            <dl>
              <div><dt>Landmarks</dt><dd>--</dd></div>
              <div><dt>Map Coverage</dt><dd>--</dd></div>
            </dl>
          </section>

          <section className="diagnostic-group muted">
            <h3>Gaussian</h3>
            <dl>
              <div><dt>Gaussian Count</dt><dd>--</dd></div>
              <div><dt>Optimizer FPS</dt><dd>--</dd></div>
            </dl>
          </section>
        </aside>
      </div>
    </main>
  );
}

