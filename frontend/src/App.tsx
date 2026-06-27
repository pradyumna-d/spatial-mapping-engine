import { useEffect, useState } from "react";

type FrameInfo = {
  frame_id: number;
  width: number;
  height: number;
  fps: number;
};

const backend =
  import.meta.env.VITE_BACKEND_URL ??
  `ws://${window.location.hostname}:8000/ws`;

export default function App() {
  const [status, setStatus] = useState("connecting");
  const [message, setMessage] = useState("");
  const [frame, setFrame] = useState<FrameInfo | null>(null);
  const [imageUrl, setImageUrl] = useState("");

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
        } else {
          pendingFrame = data;
          setFrame(data);
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

  return (
    <main>
      <h1>Live camera</h1>
      <section className="stats">
        <span className={`status ${status}`}>{status}</span>
        <span>FPS: {frame?.fps.toFixed(1) ?? "0.0"}</span>
        <span>Frame: {frame?.frame_id ?? "—"}</span>
        <span>
          Resolution: {frame ? `${frame.width} × ${frame.height}` : "—"}
        </span>
      </section>
      {imageUrl ? (
        <img src={imageUrl} alt="Live RTSP camera preview" />
      ) : (
        <div className="empty">{message || "Waiting for camera frames…"}</div>
      )}
    </main>
  );
}
