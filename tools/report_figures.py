"""Render report figures and optional time-cursor films from archived measurements."""
import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from report_statistics import DOC, read, grouped

FIG = DOC / "report_figures"
MEDIA = DOC / "report_media"
ART = Path("artifacts")
COLORS = dict(A="#315475", B="#c05514", C="#007d77", D="#86549b")
STYLES = dict(A="-", B="--", C="-.", D=":")
LABEL = dict(A="Ordinary", B="Force", C="Matched", D="Calibrated")
plt.rcParams.update({"font.size":9.5, "axes.titlesize":11, "axes.titlelocation":"left",
                     "axes.titleweight":"semibold", "axes.spines.top":False,
                     "axes.spines.right":False, "axes.grid":True, "axes.axisbelow":True,
                     "grid.alpha":.16, "legend.frameon":False, "svg.fonttype":"none",
                     "svg.hashsalt":"unifp-mpc-report", "pdf.fonttype":42, "savefig.facecolor":"white","font.family":"DejaVu Sans"})


def save(fig,name):
    for ext in ("svg","pdf","png"):
        metadata = {"Date": None} if ext == "svg" else {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(FIG/f"{name}.{ext}", dpi=175, metadata=metadata)
    plt.close(fig)


def layout(rows,cols,size):
    return plt.subplots(rows,cols,figsize=size,layout="constrained",squeeze=False)


def trace(ax, y, arm, **kwargs):
    ax.plot(.02*np.arange(1,len(y)+1),y,STYLES[arm],color=COLORS[arm],lw=1.45,
            label=LABEL[arm],**kwargs)


def architecture():
    fig,ax=plt.subplots(figsize=(10,4.1),layout="constrained")
    ax.set(xlim=(0,10),ylim=(0,4.6))
    ax.axis("off")
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.05,rounding_size=.08",
                                    facecolor=color,edgecolor="#677989",lw=1))
        ax.text(x+.12,y+h-.22,title,va="top",weight="bold",fontsize=11)
        ax.text(x+.12,y+h-.55,body,va="top",linespacing=1.38)
    def arrow(a,b):
        ax.annotate("",b,a,arrowprops=dict(arrowstyle="-|>",color="#355363",lw=1.6))
    box(.1,2.65,2.52,1.4,"Executed state","MuJoCo state, history, gait\nprevious action, drive queues\nSnapshot at each 100 ms replan","#e6f1f1")
    box(3.05,2.65,2.78,1.4,"Feasible command knots","Shift winner and elite mean\nSample EE XYZ and body rates\nOptional virtual force or offset","#f5efe3")
    box(6.27,2.65,3.56,1.4,"Parallel prediction worlds","Restore actual state into each lane\nProject request at predicted pre-action pose\nB issues force; C encodes reachable offset","#eaf0f6")
    box(6.27,.52,3.56,1.42,"Policy inside each rollout","32 × 79 history → encoder + actor\n19 joint actions → delayed, limited PD\n4 × 5 ms dynamics and solved contacts","#eaf0f6")
    box(3.05,.52,2.78,1.42,"Rank candidate outcomes","Task cost and validity tests\nFit elites; select best evaluated\ntrajectory with a finite cost","#f5efe3")
    box(.1,.52,2.52,1.42,"Live execution","Run first five policy ticks\nMeasure motion and contact\nExecution physics can differ","#e6f1f1")
    for a,b in [((2.68,3.35),(2.99,3.35)),((5.89,3.35),(6.21,3.35)),((8,2.58),(8,2.02)),
                ((6.21,1.22),(5.90,1.22)),((2.99,1.22),(2.68,1.22)),((1.35,2),(1.35,2.59))]:
        arrow(a,b)
    save(fig,"architecture")


def outcomes():
    allrows = read("final_trials")
    arms = read("report_data/mpc_groups")
    fig, axes = layout(2,3,(10,5.6))
    for j,task in enumerate(("box","cone","tether")):
        modes = ["A","Bz","Cz","Bxyz","Cxyz"] if task=="cone" else list("ABC")
        rows=[r for r in allrows if r["block"]=="general" and r["task"]==task]
        ax,bx = axes[:,j]
        for i,k in enumerate(modes):
            r=next(r for r in arms if r["block"]=="general" and r["task"]==task and r["arm"]==k)
            y,lo,hi = [100*float(r[x]) for x in ("rate","scene_ci_low","scene_ci_high")]
            ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt="o",capsize=4,color=COLORS[k[0]])
        ax.set(title=task.capitalize(),ylim=(-3,103),xticks=range(len(modes)),xticklabels=modes)
        # Each connected row is a physical scene, averaged over its two optimizer seeds.
        for case,rs in grouped(rows,["scene_seed"]).items():
            vals=[np.mean([float(r["final_error"]) for r in rs if r["arm"]==k])
                  * (1 if task=="cone" else 1000) for k in modes]
            bx.plot(modes,vals,"o-",ms=2.7,color="#607c90",alpha=.42,lw=.8)
        bx.set(ylabel="Final tilt (°)" if task=="cone" else "Final XY error (mm)")
    axes[0,0].set_ylabel("Strict success (%) · 95% interval")
    save(fig,"general_outcomes")


def pressure():
    fig,ax=layout(2,3,(10,5.3))
    goals=read("supplement_push")+read("supplement_push_heavy")
    for j,(cohort,title) in enumerate([("replay","Original · 1 kg"),("held","Fresh · 1 kg"),
                                       ("heavy_held","Same fresh placements · 1.5 kg")]):
        rs=[r for r in goals if r["cohort"]==cohort]
        arms=sorted(set(r["arm"] for r in rs))
        for case,grp in grouped(rs,["case"]).items():
            by={r["arm"]:float(r["final_error"])*1000 for r in grp}
            ax[0,j].plot(arms,[by[a] for a in arms],"o-",color="#758690",ms=2.8,lw=.8,alpha=.55)
        ax[0,j].axhline(15,ls=":",color="black")
        ax[0,j].set(title=title,ylim=(0,62))
        score=read("report_data/push_distributions")
        for i,a in enumerate(arms):
            rec=next(r for r in score if (r["arm"],r["cohort"],r["metric"])==(a,cohort,"final_error"))
            ax[1,j].bar(i,int(rec["inside_final_25"]),.63,edgecolor=COLORS[a],facecolor="none",
                        hatch="//",label="Within radius" if i==0 else None)
            ax[1,j].bar(i,int(rec["end_settled"]),.36,color=COLORS[a],label="Also slow" if i==0 else None)
        ax[1,j].set(xticks=range(len(arms)),xticklabels=arms,ylim=(0,12.3),yticks=[0,3,6,9,12])
    ax[0,0].set_ylabel("Endpoint XY error (mm)")
    ax[1,0].set_ylabel("Final hold (of 12 placements)")
    ax[1,0].legend(fontsize=8)
    save(fig,"goal_settling")


def mechanisms():
    fig,axes=layout(2,3,(11,5.6))
    base=ART/"supplemental/push/replay/4101"
    for arm in "ABC":
        with np.load(base/(arm+".npz")) as e:
            fx=e["gripper_on_box_N"]
            curves=[1000*e["ee_world"][:,0],1000*np.linalg.norm(e["effective_tracking_post"],axis=1),
                    np.cumsum(fx[:,0])*.02,1000*(e["qpos"][:,-7]-e["goal_xy"][0]),
                    np.linalg.norm(e["qvel"][:,-6:-4],axis=1),np.cumsum(np.abs(fx[:,1]))*.02]
            for ax,y in zip(axes.flat,curves,strict=True):
                trace(ax,y,arm)
            if arm=="B":
                for k,ls in [("effective_pre",":"),("nominal_pre","--")]:
                    axes[0,0].plot(np.arange(1,151)*.02,1000*e[k][:,0],ls,color="#302d2a",lw=1,
                                   label="B "+k.split("_")[0])
    for ax,lab in zip(axes.flat,["World finger X (mm)","Effective-target error (mm)","Forward impulse (N·s)",
                                 "Box X relative to goal (mm)","Box XY speed (m/s)","Absolute Y impulse (N·s)"]):
        ax.set_ylabel(lab)
    axes[0,0].legend(fontsize=7,ncol=2)
    axes[1,0].axhspan(-15,15,color="gray",alpha=.12)
    axes[1,1].axhline(.02,color="gray",ls=":")
    for ax in axes[-1]:
        ax.set_xlabel("Time from branch (s)")
    save(fig,"push_delivery")


def costs():
    fig,axes=layout(2,2,(8.8,5.6))
    rates=read("report_data/mpc_groups")
    groups=read("final_groups")
    extra=read("supplement_runtime_groups")
    for j,task in enumerate(["box","cone"]):
        for i,(group,arm,label) in enumerate([("general","A","UniFP A"),("sumo","native","Native"),
                                        ("sumo","work","Equal work"),("supplement_runtime","runtime_full","Runtime")]):
            r=next(r for r in rates if (r["block"],r["arm"],r["task"])==(group,arm,task))
            g=next(t for t in extra if t["task"]==task) if i==3 else next(
                t for t in groups if (t["block"],t["arm"],t["task"])==(group,arm,task))
            n=int(r["episodes"])
            clean=int(g["clean"] if i==3 else g["safe_success"])
            col="#315475" if i==0 else "#845988"
            axes[0,j].bar(i,100*float(r["rate"]),.65,edgecolor=col,facecolor="none",lw=1.5)
            axes[0,j].bar(i,100*clean/n,.38,color=col)
            axes[1,j].vlines(i,float(g["p50_ms"]),float(g["max_ms"]),color=col)
            axes[1,j].plot(i,float(g["p95_ms"]),"o",color=col)
        for ax in axes[:,j]:
            ax.set_xticks(range(4),["UniFP A","Native","Equal work","Runtime"],rotation=12)
        axes[0,j].set(title=task.capitalize(),ylim=(0,102))
        axes[1,j].axhline(100,color="grey",ls=":")
    axes[0,0].set_ylabel("Strict (outline) / clean (fill), %")
    axes[1,0].set_ylabel("Plan latency (ms)\nMedian–max; dot = p95")
    save(fig,"cross_system")


def uncertainty():
    fig,ax=layout(2,2,(8.8,5.6))
    for j,task in enumerate(["box","cone"]):
        group=[r for r in read("uncertainty_prediction") if r["mismatch"]=="True" and r["task"]==task]
        t=[float(r["ms"]) for r in group]
        for i,(k,m,unit) in enumerate([("obj_xy_mm","loaded_p95_xy_mm","Object XY error (mm)"),
                                       ("force_vector_err_N","loaded_p95_force_N","Hand force error (N)")]):
            ax[i,j].plot(t,[float(r[k]) for r in group],"o-",color="#315475",label="Mean, all forecasts")
            ax[i,j].plot(t,[float(r[m]) for r in group],"s--",color="#b95c18",label="p95, loaded forecasts")
            ax[i,j].set(ylabel=unit,ylim=(0,None))
        ax[0,j].set_title(task.capitalize())
        ax[1,j].set_xlabel("Lookahead (ms)")
    ax[0,0].legend(fontsize=8)
    save(fig,"forecast_error")


def task_traces():
    fig,ax=layout(2,3,(10.6,5.6))
    for arm in "ABC":
        with np.load(ART/f"final/3015/311-known-{arm}/episode.npz") as e:
            goal=e["goal_xy"]
            xy=e["qpos"][49:,-7:-5]-goal
            ax[0,0].plot(xy[:,0],xy[:,1],STYLES[arm],color=COLORS[arm],label=LABEL[arm])
        with np.load(ART/f"final/3212/311-known-{arm}/episode.npz") as e:
            trace(ax[1,0],e["tension_N"].mean(1),arm)
        k=arm if arm=="A" else arm+"z"
        with np.load(ART/f"final/3109/311-known-{k}/episode.npz") as e:
            q=e["qpos"][50:]
            angle=np.degrees(np.arccos(np.clip(1-2*(q[:,-3]**2+q[:,-2]**2),-1,1)))
            trace(ax[0,1],angle,arm)
            trace(ax[1,1],np.linalg.norm(e["qvel"][50:,-3:],axis=1),arm)
        with np.load(ART/f"final/3410/311-blind-{k}/episode.npz") as e:
            trace(ax[0,2],e["ground_on_cone_N"][50:,2],arm)
            trace(ax[1,2],np.linalg.norm(e["gripper_on_cone_N"][50:],axis=1),arm)
    ax[0,0].plot(0,0,"k+",ms=11)
    ax[0,0].set(xlabel="World X from goal (m)",ylabel="World Y from goal (m)",aspect="equal",title="Box 3015 · target at +")
    ax[0,0].legend(fontsize=8)
    ax[0,1].set(title="Cone 3109",ylabel="Tilt (°)")
    ax[0,1].axhline(10,ls=":",color="gray")
    ax[0,2].set(title="Heavy cone 3410 · blind",ylabel="Vertical ground support (N)")
    for a,lab in zip(ax[-1],["Cable tension (N)","Angular speed (rad/s)","Hand resultant (N)"]):
        a.set(xlabel="Control time (s)",ylabel=lab)
    ax[1,0].set_title("Tether 3212")
    save(fig,"task_traces")


def secondary():
    fig,ax=layout(2,2,(9,5.4))
    for arm,mode in zip("ABC",("off","conditioned","matched")):
        with np.load(ART/f"uncertainty/wall/9/{mode}/episode.npz") as e:
            trace(ax[0,0],e["wall_normal_N"].mean(1),arm)
            trace(ax[1,0],e["ee_world"][:,1]-e["ee_world"][0,1],arm)
            if arm=="B":
                ax[1,0].plot(.02*np.arange(1,len(e["qpos"])+1),e["qpos"][:,1]-e["qpos"][0,1],
                             ":",color="black",label="B base")
        with np.load(ART/f"base_load/held_pulse/{mode}/episode.npz") as e:
            trace(ax[1,1],100*np.linalg.norm(e["qpos"][:,:2]-e["reference_xy"],axis=1),arm)
            if arm=="B":
                for i,c in enumerate(["#344e65","#b7521b"]):
                    ax[0,1].step(.02*np.arange(1,len(e["qpos"])+1),e["load_world_N"][:,i],
                                 where="pre",color=c,label=["World X","World Y"][i])
    ax[0,0].axhline(15,color="black",ls=":",label="15 N target")
    for a,title,y in zip(ax.flat,["Sliding · placement 91","Unexpected pulse","Tangential travel","Locomotion"],
                         ["Normal load (N)","External torso load (N)","World Y change (m)","Base path error (cm)"]):
        a.set(title=title,ylabel=y)
    ax[0,1].legend(); ax[1,0].legend(fontsize=8,ncol=2)
    for a in ax[-1]:
        a.set_xlabel("Control time (s)")
    save(fig,"secondary")


def raw_series(family):
    # Each film compares treatments on one physical condition, with true end-tick measurements.
    folder=ART/"fixed_policy_loads/held"
    axes_labels = {"tether":["Speed along walk (m/s)","Cable tension (N)","Box progress (cm)"],
                   "payload":["Standing hand height (cm)","Walking hand height (cm)","Walking sag vs own twin (mm)"],
                   "pulse":["Constant command · drop (mm)","Synchronized · drop (mm)","Scheduled world / yaw Z force (N)"],
                   "wall":["Normal wall load (N)","Finger X (m)","Issued virtual X (N)"]}
    fig,axs=layout(1,3,(10,3.05))
    for arm in "ABC":
        if family=="tether":
            with np.load(folder/f"tether_100_{arm}{'+0' if arm=='A' else '-16'}.npz") as t:
                vec=t["command"][:2]
                ys=[t["body_velocity"]@(vec/np.linalg.norm(vec)),t["tension_N"].mean(1),100*t["box_progress"]]
        elif family=="payload":
            ys=[]
            for task in ("standing","walking"):
                with np.load(folder/f"{task}_payload_100_{arm}{'+0' if arm=='A' else '+2'}.npz") as t:
                    ys.append(100*t["finger"][:,2])
                    if task=="walking":
                        ys.append(1000*(t["unloaded_z"]-t["finger"][:,2]))
        elif family=="pulse":
            ys=[]
            for path in (folder/f"pulse_100_{arm}.npz",MEDIA/f"pulse_4201_{arm}.npz"):
                with np.load(path) as t:
                    ys.append(1000*(t["unperturbed"]-t["z"]))
            cmd=np.zeros(150)
            if arm=="B":
                cmd[40:60]=2
            ys.append(cmd)
        else:
            with np.load(ART/f"supplemental/wall/5201/{arm}.npz") as t:
                ys=[t["wall_normal_N"].mean(1),t["ee_world"][:,0],t["policy_command"][:,6]]
        for a,y in zip(axs.flat,ys):
            trace(a,y,arm)
    for a,lab in zip(axs.flat,axes_labels[family]):
        a.set(xlabel="Time after branch (s)",ylabel=lab,xlim=(0,3))
        if family=="pulse":
            a.axvspan(.8,1.2,color=".3",alpha=.10)
    if family == "pulse":
        row = next(r for r in read("final_pulse") if r["case"] == "4201" and r["mode"] == "B")
        applied = [0., float(row["physical_fz_N"]), 0.]
        axs[0,2].stairs(applied, [0., .8, 1.2, 3.], color="black", lw=1.3,
                        baseline=None, label="Physical on finger")
        axs[0,2].legend(fontsize=7, loc="lower right", ncol=2)
    axs[0,0].legend(fontsize=8)
    titles=dict(tether="Fixed-command towing · condition 100 · −16 N or equivalent velocity",
                payload="Rigid finger payload · condition 100 · +2 N or equivalent height",
                pulse="Downward physical finger pulse · constant (100) and synchronized (4201) commands",
                wall="Fixed-command wall contact · placement 5201 · +10 N ramp or equivalent position")
    fig.suptitle(titles[family],fontsize=12)
    return fig


def movie(fig,name,ffmpeg):
    cursors=[a.axvline(.02,color="#242a2c",lw=1) for a in fig.axes]
    fig.set_dpi(110)
    fig.canvas.draw()
    width,height=fig.canvas.get_width_height()
    cmd=[ffmpeg,"-v","error","-y","-f","rawvideo","-pix_fmt","rgba","-s",f"{width}x{height}",
         "-r","25","-i","-","-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2","-c:v","libx264","-threads","2",
         "-pix_fmt","yuv420p","-movflags","+faststart",str(MEDIA/(name+".mp4"))]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as w:
        for tick in range(2,151,2):
            for c in cursors:
                c.set_xdata([.02*tick]*2)
            fig.canvas.draw()
            w.stdin.write(fig.canvas.buffer_rgba())
        w.stdin.close()
        if w.wait():
            raise RuntimeError("Encoding failed")
    for c in cursors:
        c.remove()



def failures():
    fig,ax=layout(1,3,(11,3.3))
    rs=[r for r in read("exploration_trials") if r["scene"]=="82" and r["sampler"]=="local"
        and r["treatment"]=="conditioned-z" and 101<=int(r["cem"])<=110]
    for ok,shape in [(True,"o"),(False,"x")]:
        pts=[r for r in rs if (r["status"]=="success")==ok]
        ax[0,0].scatter([float(r["axial_cm"]) for r in pts],[float(r["support_moment_Nms"]) for r in pts],
                        color=COLORS["B"],marker=shape,label="Release" if ok else "Failure")
    ax[0,0].axhline(0,lw=.7,color="gray")
    ax[0,0].set(xlabel="Axial contact location (cm)",ylabel="Signed angular impulse (N·m·s)",title="Outward cone 82 · vertical force")
    ax[0,0].legend()
    with np.load(ART/"general_box/baseline/general62/episode.npz") as e:
        q=e["qpos"][49:]
        start=q[0,-7:-5]
        u=e["goal_xy"]-start
        length=np.linalg.norm(u)
        u=u/length
        xy=q[:,-7:-5]-start
        ax[0,1].plot(100*xy@u,100*xy@[-u[1],u[0]],color=COLORS["A"])
        capped = np.isclose(e["policy_command"][49:,3], .55, atol=1e-6)
        ax[0,1].scatter(100*xy[capped]@u, 100*xy[capped]@[-u[1],u[0]], s=3, alpha=.5,
                        color="#aa5415", label="EE at radius cap")
        ax[0,1].scatter(100*length,0,marker="+",s=110,color="black",label="Goal")
        ax[0,1].legend()
        ax[0,1].set(title="Oblique box 62 · timeout",xlabel="Downrange (cm)",ylabel="Cross-track (cm)")
    with np.load(ART/"tow_1s/side/off/episode.npz") as e:
        ax[0,2].step(.1*np.arange(len(e["valid_count"])),e["valid_count"],color=COLORS["A"],where="post")
        ax[0,2].set(ylim=(-.5,25),yticks=[0,8,16,24],title="65° tow · stopped",xlabel="Replan time (s)",ylabel="Eligible candidates / 24")
    save(fig,"contact_failures")


def cross_metrics():
    fig,axs=layout(2,3,(10,5.4))
    for i,task in enumerate(["box","cone"]):
        for j,(k,label) in enumerate([("final_error","Endpoint error (mm)" if i==0 else "End tilt (°)"),
                                      ("settled_s","Completion time (s)"),("hand_impulse_Ns","Hand impulse (N·s)")]):
            ax=axs[i,j]
            for n,arm in enumerate(["A","native","work","runtime_full"]):
                table=read("supplement_runtime") if n==3 else read("final_trials")
                rs=[r for r in table if r["task"]==task and (n==3 or r["arm"]==arm and r["block"] in ("general","sumo")) and r[k]]
                y=[float(r[k])*(1000 if i==0 and j==0 else 1) for r in rs]
                x=n+np.linspace(-.16,.16,len(y))
                ax.scatter(x,y,s=9,color=COLORS["A"] if n==0 else "#855586",alpha=.52)
                ax.plot([n-.19,n+.19],[np.median(y)]*2,color="black",lw=1.5)
            ax.set(xticks=range(4),xticklabels=["UniFP A","Native","Work","Runtime"],ylabel=label,title=task.capitalize())
            ax.tick_params(axis="x",labelrotation=10)
    save(fig,"system_endpoints")


def challenge_and_work():
    fig, axes = layout(2, 2, (10, 6.6))
    trials = read("final_trials")
    groups = read("final_groups")
    runtime = read("supplement_runtime_groups")
    allocations = [("A", "UniFP", "o"), ("native", "Native", "^"),
                   ("work", "Equal work", "s"), ("runtime_full", "Runtime match", "D")]
    for column, task in enumerate(["box", "cone"]):
        ax, work = axes[:, column]
        modes = ["A", "B", "C"] if task == "box" else ["A", "Bz", "Cz"]
        source = [r for r in trials if r["task"] == task and r["block"] == "challenge"]
        scene_groups = grouped(source, ["scene_seed"])
        for i, arm in enumerate(modes):
            for j, rows in enumerate(scene_groups.values()):
                values = [np.mean([float(r["final_error"]) for r in rows if r["arm"] == arm
                                   and r["physics"] == model]) for model in ("blind", "informed")]
                y = np.array(values) * (1000 if task == "box" else 1)
                x = i + np.array([-.18, .18]) + (j-2.5)*.011
                ax.plot(x, y, color=COLORS[arm[0]], alpha=.6, lw=.9)
                ax.scatter(x[0], y[0], s=19, facecolor="white", edgecolor=COLORS[arm[0]])
                ax.scatter(x[1], y[1], s=18, color=COLORS[arm[0]])
        ax.set(xticks=range(3), xticklabels=modes, title=f"Heavy {task} · six paired scenes",
               ylabel="Goal error (mm)" if task == "box" else "Final tilt (°)")
        for mode, name, marker in allocations:
            if mode == "runtime_full":
                row = next(r for r in runtime if r["task"] == task)
                calls = int(row["calls"])
            else:
                block = "general" if mode == "A" else "sumo"
                row = next(r for r in groups if r["task"] == task and r["block"] == block and r["arm"] == mode)
                calls = int(row["completed_calls"])
            work.scatter(calls/int(row["updates"]), float(row["p95_ms"]), marker=marker,
                         color=COLORS["A"] if mode == "A" else "#824f7d", s=65, label=name)
        work.set(xlabel="Completed policy evaluations / update", ylabel="Planning p95 (ms)",
                 ylim=(0, 85), title=f"Common {task} · actual work")
        work.legend(fontsize=8, loc="upper left", ncol=2)
    save(fig, "model_and_budget")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--movies",action="store_true")
    p.add_argument("--ffmpeg",default="ffmpeg")
    args=p.parse_args()
    FIG.mkdir(exist_ok=True)
    for draw in (architecture,outcomes,pressure,mechanisms,costs,uncertainty,task_traces,secondary,failures,cross_metrics,challenge_and_work):
        draw()
    for family in ("tether","payload","pulse","wall"):
        fig=raw_series(family)
        if args.movies:
            movie(fig,"fixed_"+family,args.ffmpeg)
        save(fig,"fixed_"+family)
