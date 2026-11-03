import { useEffect, useRef, useState, type RefObject } from "react";

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

type Options = {
  playing: boolean;
  racing: boolean;
  activeMode: string;
  worldId: number | undefined;
  surface: RefObject<HTMLDivElement | null>;
  send: (name: string, values: Record<string, unknown>) => Promise<unknown>;
  onError: (message: string) => void;
};

export function useGameControls({
  playing,
  racing,
  activeMode,
  worldId,
  surface,
  send,
  onError,
}: Options) {
  const [pressed, setPressed] = useState<string[]>([]);
  const [looking, setLooking] = useState(false);
  const [locked, setLocked] = useState(false);
  const controlInput = useRef<(code: string, down: boolean) => void>(() => {});
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
            ...(nextWeapon === undefined ? {} : { weapon: nextWeapon }),
          });
          if (reply === undefined) {
            dirty = false;
            break;
          }
          if (disposed) break;
          if (!racing && (dx || dy)) await send("look", { dx, dy });
          if (disposed) break;
          for (const action of pulses) {
            if (disposed) break;
            await send("action", { action });
          }
        }
      } catch (e) {
        onError(String(e));
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
  }, [playing, racing, activeMode, worldId, send, onError, surface]);

  return { pressed, looking, locked, controlInput, bindings };
}
