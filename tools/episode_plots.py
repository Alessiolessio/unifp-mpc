"""Offline component telemetry for the native 50 Hz episode format."""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from cem import DT, PREFIX
from kinematics import angle_deg, nominal_targets, world_ee_target, world_force, yaw_of


def yaw_rotate(v, pose, inverse=False):
    yaw = yaw_of(pose[..., 3:7])
    c, s = np.cos(yaw), np.sin(-yaw if inverse else yaw)
    x, y, z = np.moveaxis(v, -1, 0)
    return np.stack((c*x-s*y, s*x+c*y, z), -1)


def signals(data, first, summary, raw=None):
    q, v, cmd = (data[k] for k in ("qpos", "qvel", "policy_command"))
    root = np.concatenate((first[..., None, :7], q[..., :-1, :7]), axis=-2)
    task = summary.get("task", "box")
    issued_pos = world_ee_target(root, cmd[..., 3:6])
    issued_force = world_force(root, cmd)
    nominal = (np.full_like(issued_pos, np.nan) if summary.get("ee_force") == "matched"
               else issued_pos)
    out = {}
    if raw is not None and raw.shape[-1] != 12:
        out["request"] = raw[..., :3]
        if summary.get("ee_force") == "matched":
            nominal = nominal_targets(root, out["request"])
        if raw.shape[-1] == 9:
            out["proposal"] = yaw_rotate(raw[..., 6:9], root)
        elif raw.shape[-1] == 8:
            out["proposal"] = np.concatenate((raw[..., 6:8], np.zeros_like(raw[..., :1])), -1)
    out.update(nominal=nominal, effective=issued_pos+issued_force/200, issued=issued_force,
               command=cmd[..., :3], virtual=yaw_rotate(cmd[..., 9:12], root))
    if "ee_world" in data:
        out["finger"] = data["ee_world"]
    # MuJoCo free-joint linear velocity is world-frame; angular velocity is body-frame.
    xyz, w = -q[..., 4:7], q[..., 3:4]
    velocity = v[..., :3] + 2*np.cross(xyz, np.cross(xyz, v[..., :3]) + w*v[..., :3])
    out["measured"] = np.concatenate((velocity[..., :2], v[..., 5:6]), -1)
    contact_key = "gripper_on_" + ("box" if task == "tether" else task) + "_N"
    if contact_key in data:
        out["on object"] = data[contact_key]
    if task == "cone":
        out.update(tilt=angle_deg(q[..., -4:]), speed=np.linalg.norm(v[..., -3:], axis=-1),
                   support=data["ground_on_cone_N"][..., 2], duty=data["contact_duty"])
    if task == "wall":
        out["normal"] = data["wall_normal_N"].mean(-1)
        out["setpoint"] = np.full_like(out["normal"], summary["desired_N"])
    if task in ("box", "tether"):
        out["object"] = q[..., -7:-5]
        out["error"] = np.linalg.norm(out["object"] - summary["goal_xy_m"], axis=-1)
    if task == "tether":
        cable = data["eye_to_hitch_m"]
        out["tension"] = data["tension_N"].mean(-1)
        out["cable load"] = -out["tension"][..., None]*cable/np.linalg.norm(cable, axis=-1, keepdims=True)
    if task == "base_load":
        f = data["load_world_N"]
        out["torso load"] = np.concatenate((f, np.zeros_like(f[..., :1])), -1)
        vel = cmd[..., :2] + (cmd[..., 9:11]+yaw_rotate(out["torso load"], root, True)[..., :2])/200
        moving = (abs(vel) > .1).any(-1) | (abs(cmd[..., 2]) > .2)
        out["vel target"] = vel*moving[..., None]
        if "reference_xy" in data:
            out["path error"] = np.linalg.norm(q[..., :2]-data["reference_xy"], axis=-1)
    return out


# All other signals are post-action states or interval measurements.
PRE = {"request", "nominal", "effective", "proposal", "issued", "command", "virtual", "vel target"}
COLORS = {"request":"#168cac", "nominal":"#9352a0", "effective":"#238655", "finger":"#d66a17",
          "proposal":"#168cac", "issued":"#9352a0", "on object":"#c75e13", "command":"#167f9b",
          "measured":"#c75e13", "vel target":"#7c53aa", "virtual":"#9352a0",
          "torso load":"#bd5720", "cable load":"#bd5720", "setpoint":"#9d57a5"}


class Dashboard:
    def __init__(self, folder, e, summary, spawn_root):
        self.task = task = summary.get("task", "box")
        self.status = summary.get("status", "success" if summary.get("success") else "failed")
        n = len(e["qpos"])
        self.t = (np.arange(n)+1)*DT
        warm = 50 if task in ("box","cone") else 0
        self.warm = warm
        self.lat = e.get("planning_s", e.get("latency_s", []))
        raw = None
        box_plans = folder / "plans.npz"
        if task == "box" and summary.get("controller") in ("mpc", "mobile") and box_plans.exists():
            with np.load(box_plans) as p:
                self.lat = p["planning_s"]
                raw = p["requested_prefix_m"]
                if "requested_force_xy_N" in p:
                    raw = np.concatenate((raw, np.zeros_like(raw), p["requested_force_xy_N"]), -1)
        elif "requests" in e:
            raw = e["requests"]
        if raw is not None and not raw.size:
            raw = None
        if raw is not None:
            raw = raw.reshape(-1, raw.shape[-1])
            raw = np.pad(raw.astype(float), ((warm,0),(0,0)), constant_values=np.nan)
        # Old tether/walk files omit their settled root. Leave their first pre-action sample missing.
        first = e.get("initial_qpos", spawn_root if warm else np.full(7, np.nan))
        self.hist = signals(e, first, summary, raw)
        if warm:
            cmd = e["policy_command"][:warm]
            root = np.vstack((spawn_root[:7], e["qpos"][:warm-1, :7]))
            self.hist["nominal"][:warm] = world_ee_target(root, cmd[:, 3:6])
        self.pred = {}
        self.starts = np.array([], dtype=int)
        if "forecast_start_tick" in e:
            self.starts = e["forecast_start_tick"]
            f = {k[9:]: e[k] for k in e if k.startswith("forecast_")}
            self.pred = signals(f, f["pre_root"], summary, f["request"])
        elif ("targets" in e and e["targets"].size) or ("paths" in e and e["paths"].size):
            raw_path, issued_path, finger = ((e["targets"], e.get("projected"), e["fingers"])
                    if "targets" in e else np.moveaxis(e["paths"], 1, 0))
            self.pred = {"request":raw_path, "finger":finger}
            if issued_path is not None:
                self.pred["effective" if summary.get("ee_force") == "matched" else "nominal"] = issued_path
            self.starts = warm+PREFIX*np.arange(len(finger))

        fig = self.fig = Figure(figsize=(16,10), dpi=100, facecolor="white")
        self.canvas = FigureCanvasAgg(fig)
        self.lines, self.markers = [], []
        case = summary.get("scenario", {})
        tag = case.get("scenario_seed", summary.get("case", ""))
        mode = summary.get("ee_force", summary.get("controller", "off"))
        axes = summary.get("force_axes", "")
        seed = summary.get("cem_seed", summary.get("seed", ""))
        fig.text(.014,.947, f"{mode} {axes}   ·   CEM {seed}" + (f"   ·   scene {tag}" if tag != "" else ""),
                 fontsize=10, color="#475569")
        self.header = fig.text(.023,.972,"",fontsize=16,weight="bold",animated=True)
        fig.text(.62,.966,"Solid   executed history       Dashed   winning forecast",fontsize=11)
        self.end = self.t[-1]
        if len(self.starts):
            self.end = max(self.end, DT*(self.starts[-1]+next(iter(self.pred.values())).shape[1]))

        def panel(rect, title, channels):
            ax = fig.add_axes(rect)
            ax.set_title(title, loc="left", fontsize=10, pad=2)
            ax.tick_params(labelsize=8, length=2, pad=1)
            ax.grid(alpha=.2)
            lo, hi = np.inf, -np.inf
            for name, component, label in channels:
                for future, source in enumerate((self.hist,self.pred)):
                    if name not in source:
                        continue
                    y = source[name] if component is None else source[name][..., component]
                    if not np.isfinite(y).any():
                        continue
                    lo, hi = min(lo,np.nanmin(y)), max(hi,np.nanmax(y))
                    line, = ax.plot([], [], "--" if future else "-", lw=1.35,
                                    color=COLORS.get(name, "#216f8c" if label != "Y" else "#bc5e17"),
                                    label=label if not future else "_", animated=True)
                    self.lines.append((line, name in PRE, future, y))
            if lo != np.inf:
                margin = max(.04*(hi-lo), .02)
                ax.set_ylim(lo-margin, hi+margin)
                ax.legend(loc="upper right", fontsize=7, ncol=len(channels), framealpha=.8,
                          handlelength=1., columnspacing=.7)
            else:
                ax.text(.5,.5,"Unavailable",ha="center", transform=ax.transAxes)
            ax.set_xlim(0,self.end)
            self.markers.append(ax.axvline(0,color="#313940",alpha=.65,lw=.9,animated=True))

        task_panels = {
            "cone":[("Cone tilt · deg","tilt"),("Angular speed · rad/s","speed"),
                    ("Ground support · N","support"),("Hand contact duty · fraction","duty")],
            "box":[("Box goal error · m","error"),("World box XY · m","object")],
            "tether":[("Box goal error · m","error"),("Tendon tension · N","tension")],
            "wall":[("Wall-normal force · N","normal")],
            "base_load":[("Path error · m","path error")]}[task]
        for j,(title,key) in enumerate(task_panels):
            channels = [(key,None,key)]
            if key == "object":
                channels = [(key,a,"XY"[a]) for a in range(2)]
            elif key == "normal":
                channels += [("setpoint",None,"desired")]
            if len(task_panels) <= 2:
                rect = [.647, .802-.18*j, .327, .126]
            else:
                rect = [.64+(j%2)*.18, .81-(j//2)*.18, .158, .125]
            panel(rect,title,channels)
        rows = [
            ("EE world position", "m", ("request","nominal","effective","finger")),
            ("EE world force", "N", ("proposal","issued","on object")),
            ("Base body velocity", "m/s", ("command","vel target","measured")),
            ("Base world force", "N", ("virtual","cable load","torso load"))]
        for r,(group,unit,keys) in enumerate(rows):
            for a in range(3):
                title = f"{group}  {'XYZ'[a]} · {unit}"
                if r == 2 and a == 2:
                    title = "Base angular velocity about body Z · rad/s"
                labels = {"proposal":"virtual proposal", "issued":"actor input",
                          "on object":"hand on object", "virtual":"virtual input"}
                channels = [(k, a, labels.get(k,k)) for k in keys if k != "vel target" or a < 2]
                panel([.046+a*.327,.438-r*.134,.285,.097],title,channels)
        if task in ("tether", "wall", "base_load"):
            note = {"tether":"Torso cable load uses tick-mean tension and end-of-tick cable direction.",
                    "wall":"Positive normal is force from the gripper into the fixed wall (+world X).",
                    "base_load":"Physical torso load is the applied world wrench at the torso COM."}
            fig.text(.644,.585,note[task],fontsize=8)
        fig.text(.50,.007,"Episode time (s)",ha="center",fontsize=10)
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(fig.bbox)

    def frame(self, i, camera):
        self.canvas.restore_region(self.background)
        j = np.searchsorted(self.starts, i+1, side="right")-1
        for line, pre, future, y in self.lines:
            if future:
                if j < 0:
                    continue
                t = (self.starts[j]+np.arange(y.shape[1])+(not pre))*DT
                line.set_data(t,y[j])
            else:
                line.set_data(self.t[:i+1]-(DT if pre else 0.), y[:i+1])
            line.axes.draw_artist(line)
        for marker in self.markers:
            marker.set_xdata([self.t[i]]*2)
            marker.axes.draw_artist(marker)
        status = self.status if i == len(self.t)-1 else "warmup" if i+1 < self.warm else "executing"
        update = (i+1-self.warm)//PREFIX
        update = min(update, len(self.lat)-1)
        timing = f"   |   plan {1000*self.lat[update]:.1f} ms" if 0 <= update < len(self.lat) else ""
        self.header.set_text(f"{self.task.replace('_',' ').title()}   |   {self.t[i]:.2f} s   |   {status}{timing}")
        self.fig.draw_artist(self.header)
        rgb = np.asarray(self.canvas.buffer_rgba())[:,:,:3].copy()
        rgb[62:422,22:982] = camera[:360]
        return rgb
