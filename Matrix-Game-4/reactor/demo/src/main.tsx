import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ReactorProvider,
  ReactorView,
  useReactor,
  useReactorMessage,
} from "@reactor-team/js-sdk";
import { Logo } from "@reactor-team/ui";
import "./style.css";

const API_URL = import.meta.env.VITE_REACTOR_URL || "http://localhost:8080";
const scenes = [
  {
    id: "ancient-egypt",
    name: "Ancient Egypt",
    kind: "Third-person shooter",
    detail: "Sunlit ruins & desert combat",
  },
  {
    id: "castle",
    name: "Castle",
    kind: "Third-person shooter",
    detail: "Stone courtyards & close combat",
  },
  {
    id: "racing",
    name: "Racing",
    kind: "Driving",
    detail: "City streets, trials & pursuit",
  },
];
type KeyBinding = { code: string; key: string; action: string };
const movement: KeyBinding[] = [
  { code: "KeyW", key: "W", action: "Forward" },
  { code: "KeyA", key: "A", action: "Left" },
  { code: "KeyS", key: "S", action: "Backward" },
  { code: "KeyD", key: "D", action: "Right" },
];
const shooterKeys: KeyBinding[] = [
  ...movement,
  { code: "Mouse0", key: "LMB", action: "Fire" },
  { code: "Mouse2", key: "RMB", action: "Aim" },
  { code: "Space", key: "Space", action: "Jump" },
  { code: "ShiftLeft", key: "Shift", action: "Dash" },
  { code: "KeyF", key: "F", action: "Shield" },
  { code: "KeyC", key: "C", action: "Crouch" },
  { code: "KeyR", key: "R", action: "Reload" },
  { code: "KeyT", key: "T", action: "Trap" },
  { code: "KeyH", key: "H", action: "Cover" },
  { code: "Digit1", key: "1", action: "Pistol" },
  { code: "Digit2", key: "2", action: "Rifle" },
  { code: "Digit3", key: "3", action: "Shotgun" },
];
const raceKeys: KeyBinding[] = [
  { code: "KeyW", key: "W", action: "Accelerate" },
  { code: "KeyA", key: "A", action: "Steer left" },
  { code: "KeyS", key: "S", action: "Reverse" },
  { code: "KeyD", key: "D", action: "Steer right" },
  { code: "Space", key: "Space", action: "Handbrake" },
  { code: "KeyB", key: "B", action: "Brake" },
];
const aliases: Record<string, string> = {
  ArrowUp: "KeyW",
  ArrowDown: "KeyS",
  ArrowLeft: "KeyA",
  ArrowRight: "KeyD",
  ShiftRight: "ShiftLeft",
};

async function waitForSessionCleanup() {
  // Disconnect closes our connection before the server finishes its session
  // hooks. Keep Connect disabled until that cleanup has actually completed.
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    const response = await fetch(`${API_URL.replace(/\/$/, "")}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!response.ok)
      throw new Error(`Readiness check failed: HTTP ${response.status}`);
    if ((await response.json()).state === "available") return;
    await new Promise((resolve) => window.setTimeout(resolve, 250));
  }
  throw new Error(
    "Disconnected. Runtime is still finishing the session; reconnect when it is available.",
  );
}

type World = {
  scene: string | null;
  running: boolean;
  world_id: number;
  completed_chunks: number;
  frame_end: number;
  last_chunk_seconds: number | null;
  controls: Record<string, string | number | boolean>;
};

function Workspace() {
  const status = useReactor((s) => s.status);
  const connect = useReactor((s) => s.connect);
  const disconnect = useReactor((s) => s.disconnect);
  const send = useReactor((s) => s.sendCommand);
  const lastError = useReactor((s) => s.lastError);
  const [world, setWorld] = useState<World | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [locked, setLocked] = useState(false);
  const [seed, setSeed] = useState(0);
  const [mode, setMode] = useState("time_trial");
  const [route, setRoute] = useState("downtown");
  const [scene, setScene] = useState("ancient-egypt");
  const [pressed, setPressed] = useState<string[]>([]);
  const [looking, setLooking] = useState(false);
  const controlInput = useRef<(code: string, down: boolean) => void>(() => {});
  const commandLock = useRef(false);
  const surface = useRef<HTMLDivElement>(null);
  const ready = status === "ready";
  const playing = ready && Boolean(world?.running) && !world?.controls.paused;
  const racing = (world?.running ? world.scene : scene) === "racing";
  const activeMode = world?.running
    ? String(world.controls.active_race_mode)
    : mode;
  const bindings = racing
    ? [
        ...raceKeys,
        ...(activeMode === "time_trial"
          ? [{ code: "KeyR", key: "R", action: "Recover" }]
          : []),
        ...(activeMode !== "drive"
          ? [
              { code: "KeyP", key: "P", action: "Pause race" },
              { code: "Enter", key: "Enter", action: "Resume race" },
            ]
          : []),
      ]
    : shooterKeys;
  const selected = scenes.find((item) => item.id === scene)!;
  useReactorMessage((message) => {
    if (message.type === "state_update") setWorld(message.data as World);
  });
  useEffect(() => {
    if (status !== "ready") {
      setWorld(null);
      setPressed([]);
      setLooking(false);
    }
  }, [status]);
  useEffect(() => {
    if (typeof world?.controls.scene === "string")
      setScene(world.controls.scene);
  }, [world?.controls.scene]);

  async function perform(action: () => Promise<unknown>) {
    if (commandLock.current) return;
    commandLock.current = true;
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      commandLock.current = false;
    }
  }
  async function command(name: string, values: Record<string, unknown> = {}) {
    const response = await send(name, values);
    if (response === undefined)
      throw new Error(`${name} was refused. See the command error below.`);
    if (response.type === "state_update") setWorld(response.data as World);
    return response;
  }

  useEffect(() => {
    if (!playing) {
      setPressed([]);
      setLooking(false);
      return;
    }
    const keys = new Set<string>();
    let fire = false,
      ads = false,
      mouseX = 0,
      mouseY = 0;
    let sending = false,
      dirty = false,
      disposed = false;
    const actions: string[] = [];
    let weapon: number | undefined;
    let lookTimer: number | undefined;
    const display = () =>
      setPressed([
        ...keys,
        ...(fire ? ["Mouse0"] : []),
        ...(ads ? ["Mouse2"] : []),
      ]);
    const has = (...names: string[]) => names.some((name) => keys.has(name));
    const reset = () => {
      keys.clear();
      fire = ads = false;
      mouseX = mouseY = 0;
      actions.length = 0;
      weapon = undefined;
      display();
      setLooking(false);
      window.clearTimeout(lookTimer);
    };
    // One awaited control batch at a time. Coalesce held state and mouse deltas
    // while the previous chunk runs, so slow inference cannot grow a key queue.
    async function drain() {
      dirty = true;
      if (sending) return;
      sending = true;
      try {
        while (dirty && !disposed) {
          dirty = false;
          const forward = +has("KeyW", "ArrowUp") - +has("KeyS", "ArrowDown");
          const right = +has("KeyD", "ArrowRight") - +has("KeyA", "ArrowLeft");
          const dx = mouseX,
            dy = mouseY,
            nextWeapon = weapon;
          mouseX = mouseY = 0;
          weapon = undefined;
          const pulses = actions.splice(0);
          const reply = await send("set_controls", {
            forward,
            right,
            fire,
            ads,
            shield: has("KeyF"),
            crouch: has("KeyC"),
            throttle: forward,
            steer: right,
            brake: has("KeyB") ? 1 : 0,
            handbrake: has("Space"),
          });
          if (reply === undefined) {
            dirty = false;
            break;
          }
          if (disposed) break;
          if (!racing && (dx || dy)) await send("look", { dx, dy });
          if (disposed) break;
          if (!racing && nextWeapon !== undefined)
            await send("set_weapon", { weapon: nextWeapon });
          for (const action of pulses) {
            if (disposed) break;
            await send("action", { action });
          }
        }
      } catch (e) {
        setError(String(e));
      } finally {
        sending = false;
      }
    }
    const actionKeys: Record<string, string> = racing
      ? activeMode === "drive"
        ? {}
        : {
            ...(activeMode === "time_trial" ? { KeyR: "recover" } : {}),
            KeyP: "pause",
            Enter: "resume",
          }
      : {
          Space: "jump",
          ShiftLeft: "dash",
          KeyR: "reload",
          KeyT: "trap",
          KeyH: "cover",
        };
    const allowed = new Set(bindings.map((binding) => binding.code));
    const input = (code: string, down: boolean) => {
      if (!allowed.has(code)) return;
      if (code === "Mouse0") fire = down;
      else if (code === "Mouse2") ads = down;
      else if (down) {
        if (keys.has(code)) return;
        keys.add(code);
        if (actionKeys[code] && actions.length < 8)
          actions.push(actionKeys[code]);
        if (/^Digit[123]$/.test(code)) weapon = Number(code.slice(-1)) - 1;
      } else keys.delete(code);
      display();
      void drain();
    };
    controlInput.current = input;
    const key = (e: KeyboardEvent) => {
      if (document.pointerLockElement !== surface.current) return;
      const code = aliases[e.code] || e.code;
      if (e.repeat || !allowed.has(code)) return;
      e.preventDefault();
      input(code, e.type === "keydown");
    };
    const move = (e: MouseEvent) => {
      if (!racing && document.pointerLockElement === surface.current) {
        mouseX = Math.max(-2000, Math.min(2000, mouseX + e.movementX));
        mouseY = Math.max(-2000, Math.min(2000, mouseY + e.movementY));
        setLooking(true);
        window.clearTimeout(lookTimer);
        lookTimer = window.setTimeout(() => setLooking(false), 180);
      }
    };
    const mouse = (e: MouseEvent) => {
      if (document.pointerLockElement !== surface.current) return;
      input(`Mouse${e.button}`, e.type === "mousedown");
    };
    const release = () => {
      reset();
      void drain();
    };
    const lock = () => {
      const active = document.pointerLockElement === surface.current;
      setLocked(active);
      if (!active) release();
    };
    const visibility = () => {
      if (document.hidden) release();
    };
    const timer = window.setInterval(() => {
      if (mouseX || mouseY) void drain();
    }, 16);
    document.addEventListener("keydown", key);
    document.addEventListener("keyup", key);
    document.addEventListener("mousemove", move);
    document.addEventListener("mousedown", mouse);
    document.addEventListener("mouseup", mouse);
    document.addEventListener("pointerlockchange", lock);
    document.addEventListener("visibilitychange", visibility);
    window.addEventListener("blur", release);
    return () => {
      disposed = true;
      reset();
      controlInput.current = () => {};
      window.clearInterval(timer);
      document.removeEventListener("keydown", key);
      document.removeEventListener("keyup", key);
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mousedown", mouse);
      document.removeEventListener("mouseup", mouse);
      document.removeEventListener("pointerlockchange", lock);
      document.removeEventListener("visibilitychange", visibility);
      window.removeEventListener("blur", release);
      if (document.pointerLockElement === surface.current)
        document.exitPointerLock();
      setLocked(false);
    };
  }, [playing, racing, activeMode, world?.world_id, send]);

  return (
    <main>
      <header>
        <a
          className="brand"
          href="https://reactor.inc"
          target="_blank"
          rel="noreferrer"
        >
          <Logo variant="symbol" width={24} height={24} />
          <strong>Matrix-Game 4.0</strong>
          <span>on Reactor</span>
        </a>
        <span className={`connection ${ready ? "live" : ""}`}>{status}</span>
        <button
          disabled={busy || !["ready", "disconnected"].includes(status)}
          onClick={() =>
            void perform(async () => {
              if (ready) {
                if (world?.running) await command("stop");
                await disconnect();
                setWorld(null);
                await waitForSessionCleanup();
              } else await connect();
            })
          }
        >
          {ready ? "Disconnect" : "Connect"}
        </button>
      </header>
      <div className="workspace">
        <aside>
          <p className="eyebrow">PLAY A CODE WORLD</p>
          <h1>
            One world.
            <br />
            {" "}
            Two views.
          </h1>
          <p>
            The native game and its neural render share the same video frame.
            Your controls change the next captured frames.
          </p>
          <div className="scene-heading">
            <span className="eyebrow">01 / CHOOSE A SCENE</span>
            <span>3 worlds</span>
          </div>
          <div className="scene-grid" role="group" aria-label="Scenes">
            {scenes.map((item) => (
              <button
                key={item.id}
                className={`scene-card ${scene === item.id ? "selected" : ""}`}
                aria-label={`Select ${item.name}`}
                aria-pressed={scene === item.id}
                disabled={busy || Boolean(world?.running)}
                onClick={() => setScene(item.id)}
              >
                <img
                  src={`${import.meta.env.BASE_URL}scenes/${item.id}.jpg`}
                  alt={`${item.name} generated gameplay preview`}
                />
                <span className="scene-copy">
                  <strong>{item.name}</strong>
                  <small>{item.kind}</small>
                </span>
                <span className="selected-mark" aria-hidden="true">
                  {scene === item.id ? "✓" : ""}
                </span>
              </button>
            ))}
          </div>
          <p className="hint scene-detail">
            {selected.detail}.{" "}
            {world?.running
              ? "Stop to choose another scene."
              : "Selection starts a fresh game when you press Start."}
          </p>
          <label>
            Seed
            <input
              type="number"
              min="0"
              max="2147483647"
              value={seed}
              disabled={Boolean(world?.running)}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </label>
          {racing && (
            <>
              <label>
                Race mode
                <select
                  value={mode}
                  disabled={Boolean(world?.running)}
                  onChange={(e) => {
                    setMode(e.target.value);
                    if (e.target.value === "pursuit") setRoute("avenue");
                  }}
                >
                  <option value="time_trial">Time trial</option>
                  <option value="drive">Free drive</option>
                  <option value="pursuit">Pursuit</option>
                </select>
              </label>
              <label>
                Route
                <select
                  value={route}
                  disabled={Boolean(world?.running) || mode === "pursuit"}
                  onChange={(e) => setRoute(e.target.value)}
                >
                  {["downtown", "avenue", "skyway"].map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                </select>
              </label>
            </>
          )}
          <button
            className="primary"
            disabled={!ready || busy || Boolean(world?.running)}
            onClick={() =>
              void perform(async () => {
                if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647)
                  throw new Error(
                    "Seed must be an integer between 0 and 2147483647.",
                  );
                // Generated setters declare no reply. Their success also resolves
                // undefined; only commands with a declared reply use command().
                await send("set_seed", { seed });
                await send("set_scene", { scene });
                if (racing) {
                  await send("set_race_mode", { race_mode: mode });
                  await send("set_route", { route });
                }
                await command("start");
              })
            }
          >
            {busy ? "Working…" : "Start game"}
          </button>
          <div className="row">
            <button
              disabled={!ready || busy || !world?.running}
              onClick={() =>
                void perform(() =>
                  command("set_paused", { paused: !world?.controls.paused }),
                )
              }
            >
              {world?.controls.paused ? "Resume" : "Pause"}
            </button>
            <button
              disabled={!ready || busy || !world?.running}
              onClick={() => void perform(() => command("stop"))}
            >
              Stop
            </button>
          </div>
          <div className="help">
            <h2>Controls</h2>
            <p>
              Click the video to capture your keyboard and mouse. Esc releases
              them.
            </p>
            <p>
              {racing
                ? "W/S throttle & reverse · A/D steer · Space handbrake · B brake · R recover (time trial) · P/Enter pause/resume race"
                : "WASD move · Mouse look · Left click fire · Right click aim · 1/2/3 weapons · Space jump · Shift dash · F shield · C crouch · R reload · T trap · H cover"}
            </p>
            <p>
              Release a key to release its control. Queued video still plays
              before new inputs become visible.
            </p>
          </div>
          <p className="notice">
            {locked
              ? "Input captured · Esc to release"
              : "Keyboard and mouse released"}
          </p>
          {(error || lastError) && (
            <p role="alert">{error || String(lastError)}</p>
          )}
        </aside>
        <section className="stage">
          <div className="stage-heading">
            <div>
              <p className="eyebrow">02 / ONE TIMELINE</p>
              <h2>Play the same moment.</h2>
            </div>
            <span className="tag">40 FPS · synchronized</span>
          </div>
          <div className="labels">
            <span>NATIVE DRAFT</span>
            <span>NEURAL RENDER</span>
          </div>
          <div
            className="video-shell"
            ref={surface}
            tabIndex={0}
            onClick={() => {
              if (playing)
                void surface.current
                  ?.requestPointerLock()
                  .catch((e) => setError(String(e)));
            }}
            onContextMenu={(e) => e.preventDefault()}
          >
            <ReactorView
              track="main_video"
              muted
              videoObjectFit="contain"
              className="video"
            />
            {!world?.completed_chunks && (
              <div className="placeholder">
                {world?.running
                  ? "Preparing the first generated frame…"
                  : "Connect, then Start game"}
              </div>
            )}
          </div>
          <footer>
            <span>
              {world
                ? `World ${world.world_id} · ${world.completed_chunks} chunks · ${world.frame_end + 1} frames`
                : "Awaiting a session"}
            </span>
            <span>
              {world?.last_chunk_seconds == null
                ? ""
                : `${world.last_chunk_seconds.toFixed(3)} s / chunk`}
            </span>
          </footer>
          <p className="caption">
            Each left/right pair is composed after generation. The draft does
            not run ahead of the displayed result.
          </p>
          <div className="control-heading">
            <span className="eyebrow">03 / LIVE INPUT</span>
            <span className="input-summary" role="status">
              {!playing
                ? world?.controls.paused
                  ? "Paused · controls released"
                  : "Start a game to take control"
                : [
                    ...bindings
                      .filter((binding) => pressed.includes(binding.code))
                      .map((binding) => binding.action),
                    ...(looking ? ["Look"] : []),
                  ].join(" + ") || "Controls released"}
            </span>
          </div>
          <div className="controls" role="group" aria-label="Game controls">
            {bindings.map((binding) => (
              <button
                key={binding.code}
                aria-label={binding.action}
                aria-pressed={pressed.includes(binding.code)}
                data-code={binding.code}
                className={pressed.includes(binding.code) ? "pressed" : ""}
                disabled={!playing || busy}
                onPointerDown={(e) => {
                  e.preventDefault();
                  e.currentTarget.setPointerCapture(e.pointerId);
                  controlInput.current(binding.code, true);
                }}
                onPointerUp={() => controlInput.current(binding.code, false)}
                onPointerCancel={() =>
                  controlInput.current(binding.code, false)
                }
                onLostPointerCapture={() =>
                  controlInput.current(binding.code, false)
                }
                onKeyDown={(e) => {
                  if (e.key === " " || e.key === "Enter") {
                    e.preventDefault();
                    if (!e.repeat) controlInput.current(binding.code, true);
                  }
                }}
                onKeyUp={(e) => {
                  if (e.key === " " || e.key === "Enter") {
                    e.preventDefault();
                    controlInput.current(binding.code, false);
                  }
                }}
                onBlur={() => controlInput.current(binding.code, false)}
              >
                <kbd>{binding.key}</kbd>
                <span>{binding.action}</span>
              </button>
            ))}
          </div>
          <p className="hint">
            Click the video to use your keyboard and mouse, or hold the buttons
            above. Highlights show your current input, which affects the next
            captured frames; already-buffered video plays first. Esc, Pause,
            Stop and leaving the window release held controls.
          </p>
          <div className="stage-bottom">
            <span>
              {world?.scene
                ? scenes.find((item) => item.id === world.scene)?.name
                : "Choose your world"}
              {world?.running && !racing
                ? ` · ${["Pistol", "Rifle", "Shotgun"][Number(world.controls.weapon)]}`
                : ""}
            </span>
            <span>Original game & inference · Skywork AI</span>
          </div>
        </section>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <ReactorProvider
    modelName="matrix-game-4"
    local
    apiUrl={API_URL}
    modelTracks={[{ name: "main_video", kind: "video", direction: "recvonly" }]}
    connectOptions={{ autoConnect: false }}
  >
    <Workspace />
  </ReactorProvider>,
);
