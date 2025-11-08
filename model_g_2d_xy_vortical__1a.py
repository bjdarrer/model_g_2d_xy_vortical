"""
Model G (2D proper, x–y) with Vortical Motion — SAFE + RESUMABLE + 3D SURFACE + VORTICITY

- Written by Brendan Darrer aided by ChatGPT5 date: 8th November 2025
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb and
https://github.com/frostburn/tf2-model-g with https://github.com/frostburn/tf2-model-g/blob/master/docs/overview.pdf
- with ChatGPT5 writing it and Brendan guiding it to produce a clean code.
What’s new
- Adds a compressible 2D fluid velocity field u=(ux, uy) that advects G, X, Y and couples via density ρ.
- Evolves u with a simplified compressible Navier–Stokes step (isothermal pressure, viscosity) using a pseudo‑spectral method.
- Generates true vortical motion (nonzero curl u), and visualizes it alongside the 3D surface (Y, G, X/10).
- Periodic boundaries (spectral) for stability and simple derivatives.
- Safe & resumable: segmented integration, checkpoints, MP4 assembly at the end.

References
- Coupling form (advection, ∇·u term, isothermal pressure, viscosity) follows the outline used by L. Pakkanen (2020) — see attached overview.

Install
    pip install numpy scipy matplotlib imageio imageio[ffmpeg]

Run (batch)
    python model_g_2d_xy_vortical__1a.py

Run (live viewer)
    MPLBACKEND=TkAgg python model_g_2d_xy_vortical__1a.py --live

Notes
- This version uses a pseudo‑spectral (FFT) scheme with periodic BCs. Your earlier FD/Dirichlet codes remain separate.
- For stability, keep dt modest. Start with nx=ny=192, Lx=Ly=60, dt=0.005, Tfinal=5–10.
"""
import os
import time
import math
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio
from numpy.fft import rfftn, irfftn, fftfreq, rfftfreq
from numpy.fft import rfft2, irfft2

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description='Model G 2D (xy) with vortical motion (pseudo-spectral, periodic)')
parser.add_argument('--live', action='store_true', help='Enable live 3D viewer')
parser.add_argument('--live_stride', type=int, default=5, help='Update live viewer every Nth frame (default 5)')
parser.add_argument('--downsample', type=int, default=0, help='Rendering downsample (0=auto)')
parser.add_argument('--nx', type=int, default=192)
parser.add_argument('--ny', type=int, default=192)
parser.add_argument('--Lx', type=float, default=60.0)
parser.add_argument('--Ly', type=float, default=60.0)
parser.add_argument('--Tfinal', type=float, default=10.0)
parser.add_argument('--segment_dt', type=float, default=0.5)
parser.add_argument('--dt', type=float, default=0.005, help='Time step for explicit splitting')
parser.add_argument('--nt_anim', type=int, default=200)
parser.add_argument('--max_frames', type=int, default=200, help='Cap frames for MP4 (helps long runs)')
parser.add_argument('--zlim', type=float, default=1.0, help='±Z limit for surface plot')
args = parser.parse_args()

# ---------------- Paths ----------------
run_name  = 'model_g_2d_xy_vortical__1a'
out_dir   = f'out_{run_name}'
frames_dir= os.path.join(out_dir, 'frames')
ckpt_path = os.path.join(out_dir, 'checkpoint_vortical.npz')
mp4_path  = os.path.join(out_dir, f'{run_name}.mp4')
final_png = os.path.join(out_dir, 'final_snapshot.png')
os.makedirs(frames_dir, exist_ok=True)

# ---------------- Grid & Fourier wavenumbers (periodic) ----------------
Lx, Ly = args.Lx, args.Ly
nx, ny = args.nx, args.ny
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='xy')  # shape (ny, nx)

# BJD commented out below and replaced new code after - 8.11.2025 15:18
#kx = 2*np.pi*fftfreq(nx, d=Lx/nx)          # shape (nx,)
#ky = 2*np.pi*rfftfreq(ny, d=Ly/ny)         # shape (ny//2+1,)
#KX, KY = np.meshgrid(kx, ky, indexing='xy') # spectral grids for rfft over y
#K2 = KX**2 + KY**2
#K2[0,0] = 1.0  # avoid divide-by-zero for diagnostic uses

# Wavenumbers — NOTE: match array shape (ny, nx_rfft)
kx = 2*np.pi*fftfreq(nx, d=Lx/nx)       # shape (nx,)
ky = 2*np.pi*rfftfreq(ny, d=Ly/ny)      # shape (ny//2+1,)
KX, KY = np.meshgrid(kx, ky, indexing='ij')  # corrected indexing
K2 = KX**2 + KY**2
K2[0, 0] = 1.0  # avoid divide-by-zero


# ---------------- Model G parameters (eqs17 mapping) ----------------
params = {
    'a': 14.0,
    'b': 29.0,
    'dx': 1.0,
    'dy': 12.0,
    'p': 1.0,
    'q': 1.0,
    'g': 0.1,
    's': 0.0,
    'u_cross': 0.0,     # u in paper eqs is cross-coupling; here keep 0 (not fluid u)
    'w': 0.0,
}

# Fluid coupling
rho0 = 1.0
alphaG = 0.02
alphaX = 0.02
alphaY = 0.02
cs2    = 1.0          # c_s^2 (isothermal pressure)
nu     = 0.25         # kinematic viscosity

# Seeding (same style as before, but 2D Gaussian)
def bell(s, x):
    return np.exp(-(x/s)**2 / 2.0)

Tseed = 3.0
seed_sigma_space = 2.0
seed_sigma_time  = 2.0
seed_centers = [(Lx/2, Ly/2)]  # center of box

# ---------------- Helpers: FFT, spectral derivatives ----------------
"""
def rfft2(f):
    return rfftn(f, s=(ny, nx), axes=(0,1))  # rfft over last axis? we set axes=(0,1) with shapes (ny,nx)

def irfft2(F):
    return irfftn(F, s=(ny, nx), axes=(0,1))

def grad(f):
    F = rfft2(f)
    fx = irfft2(1j*KX*F)
    fy = irfft2(1j*KY*F)
    return fx, fy

def laplacian(f):
    F = rfft2(f)
    return irfft2(-K2*F)
"""
# ---------------- FFT helpers ----------------

def rfft2c(f):
    # 2D real FFT, matching array (ny, nx)
    return np.fft.rfft2(f, s=(ny, nx))

def irfft2c(F):
    return np.fft.irfft2(F, s=(ny, nx))

def grad(f):
    F = rfft2c(f)
    fx = irfft2c(1j * KX * F)
    fy = irfft2c(1j * KY * F)
    return fx, fy

def laplacian(f):
    F = rfft2c(f)
    return irfft2c(-K2 * F)

Ftest = rfft2c(np.random.rand(ny, nx))
print("KX, KY, F shapes:", KX.shape, KY.shape, Ftest.shape)

# ---------------- Reaction terms (dimensionless eqs like eqs13) ---------------
a = params['a']; b = params['b']; p_par = params['p']; q_par = params['q']; g_par = params['g']; s_par = params['s']; w_par = params['w']; u_cross = params['u_cross']
# Homogeneous state
G0 = (a + g_par*w_par) / (q_par - g_par*p_par)
X0 = (p_par*a + q_par*w_par) / (q_par - g_par*p_par)
Y0 = ((s_par*X0**2 + b) * X0 / (X0**2 + u_cross)) if (X0**2 + u_cross)!=0 else 0.0
print(f"Homogeneous state: G0={G0:.6g}, X0={X0:.6g}, Y0={Y0:.6g}")

# Initial fields (fluctuations around homogeneous state)
pG = np.zeros((ny, nx))
pX = np.zeros((ny, nx))
pY = np.zeros((ny, nx))

# Velocity field
ux = np.zeros((ny, nx))
uy = np.zeros((ny, nx))

# Forcing chi(x,y,t)
def chi_xy_t(t):
    spatial = np.zeros((ny, nx))
    for (xc, yc) in seed_centers:
        spatial += np.exp(-(((X-xc))**2 + ((Y-yc))**2) / (2*seed_sigma_space**2))
    return -spatial * bell(seed_sigma_time, t - Tseed)

# ---------------- Advection (semi-Lagrangian, bilinear) -----------------------
# Backtrace positions and sample with bilinear interpolation under periodic BCs

def advect_scalar(phi, ux, uy, dt):
    # positions at time t - dt along velocity
    Xp = (X - dt*ux) % Lx
    Yp = (Y - dt*uy) % Ly
    # convert to fractional indices
    fx = (Xp / Lx) * nx
    fy = (Yp / Ly) * ny
    i0 = np.floor(fx).astype(int) % nx
    j0 = np.floor(fy).astype(int) % ny
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    sx = fx - np.floor(fx)
    sy = fy - np.floor(fy)
    # bilinear
    out = ((1-sx)*(1-sy)*phi[j0, i0] + sx*(1-sy)*phi[j0, i1] + (1-sx)*sy*phi[j1, i0] + sx*sy*phi[j1, i1])
    return out

# ---------------- Time stepping ----------------
Tfinal = args.Tfinal
segment_dt = args.segment_dt
dt = args.dt
nt_anim = args.nt_anim
max_frames = args.max_frames

frame_times = np.linspace(0.0, Tfinal, nt_anim)

# Checkpoint IO
os.makedirs(out_dir, exist_ok=True)

def save_ckpt(t_curr, pG, pX, pY, ux, uy, next_frame_idx, frames_done):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)  # ✅ ensure folder exists
    tmp = ckpt_path + '.tmp'
    np.savez_compressed(
        tmp,
        t_curr=t_curr, pG=pG, pX=pX, pY=pY, ux=ux, uy=uy,
        next_frame_idx=next_frame_idx,
        frames_done=np.array(sorted(list(frames_done)), dtype=np.int32)
    )
    #os.replace(tmp, ckpt_path)
    #BJD included below 8.11.2025 15:40
    try:
     os.replace(tmp, ckpt_path)
    except FileNotFoundError:
     print(f"[WARN] Temporary checkpoint {tmp} missing, skipping rename.")


def load_ckpt():
    if not os.path.exists(ckpt_path):
        return None
    d = np.load(ckpt_path, allow_pickle=True)
    return {
        't_curr': float(d['t_curr']),
        'pG': d['pG'], 'pX': d['pX'], 'pY': d['pY'], 'ux': d['ux'], 'uy': d['uy'],
        'next_frame_idx': int(d['next_frame_idx']),
        'frames_done': set(int(v) for v in d['frames_done'].tolist())
    }

# --------------- Rendering (3D surfaces + vorticity underlay) ----------------

def auto_downsample(nx, ny, user_ds):
    if user_ds and user_ds > 0:
        return max(1, int(user_ds))
    return max(1, max(nx, ny)//120)

DS = auto_downsample(nx, ny, args.downsample)

# vorticity ωz = ∂uy/∂x - ∂ux/∂y

def vorticity(ux, uy):
    uyx, uyy = grad(uy)
    uxx, uxy = grad(ux)
    return uyx - uxy


def render_surface(pG, pX, pY, ux, uy, t, fpath, zlim=1.0, live_ax=None):
    step = DS
    Xs, Ys = X[::step, ::step], Y[::step, ::step]
    pG_s = pG[::step, ::step]
    pX_s = (pX/10.0)[::step, ::step]
    pY_s = pY[::step, ::step]

    vort = vorticity(ux, uy)
    vort_s = vort[::step, ::step]

    if live_ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = live_ax
        ax.clear()

    # Optionally color the surface by vorticity magnitude on pY surface
    # Using pY_s as height; colormap from vorticity underlay
    cmap = cm.coolwarm
    norm = plt.Normalize(vmin=-np.max(np.abs(vort_s)), vmax=np.max(np.abs(vort_s)))
    facecolors = cmap(norm(vort_s))

    ax.plot_surface(Xs, Ys, pY_s, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.85)
    ax.plot_surface(Xs, Ys, pG_s, cmap='Blues',   alpha=0.5, linewidth=0)
    ax.plot_surface(Xs, Ys, pX_s, cmap='Purples', alpha=0.5, linewidth=0)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Potential')
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_zlim(-zlim, zlim)
    ax.view_init(elev=35, azim=225)
    ax.set_title(f'Model G 2D + Vortices — t={t:.2f}  (ds={step}x)')

    if live_ax is None:
        plt.tight_layout(); plt.savefig(fpath, dpi=120); plt.close()
    else:
        live_ax.figure.canvas.draw_idle()

# ---------------- Live viewer ----------------
class LiveViewer:
    def __init__(self, stride=5):
        self.enabled = args.live
        self.stride = max(1, int(stride))
        self.paused = False
        self.quit = False
        self.fig = None
        self.ax = None
        if self.enabled:
            if matplotlib.get_backend().lower() == 'agg':
                try:
                    matplotlib.use('TkAgg')
                except Exception:
                    pass
            plt.ion()
            self.fig = plt.figure(figsize=(9,6))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            self.draw_text()
    def on_key(self, event):
        if event.key == 'p':
            self.paused = not self.paused; self.draw_text()
        elif event.key == 'q':
            self.quit = True
        elif event.key == 's':
            if self.fig is not None:
                self.fig.savefig(os.path.join(out_dir, 'live_snapshot.png'), dpi=140)
                print('[Live] Saved live_snapshot.png')
    def draw_text(self):
        if not self.enabled: return
        self.ax.text2D(0.03, 0.95, '[p] pause/resume  [q] quit  [s] snapshot', transform=self.ax.transAxes)
    def maybe_update(self, frame_idx, pG, pX, pY, ux, uy, t):
        if not self.enabled: return
        if frame_idx % self.stride != 0: return
        while self.paused and not self.quit:
            plt.pause(0.1)
        if self.quit: return
        render_surface(pG, pX, pY, ux, uy, t, None, zlim=args.zlim, live_ax=self.ax)
        plt.pause(0.001)

live = LiveViewer(stride=args.live_stride)

# ---------------- Time integration (operator splitting) -----------------------
# Splitting per small dt:
#   1) Reaction/Local PDE source terms for (pG,pX,pY) with explicit RK2
#   2) Diffusion via spectral decay: J <- ifft( exp(-D*K2*dt) * fft(J) )
#   3) Velocity update u with explicit RK2 on compressible NS (pressure + viscosity + convection)
#   4) Semi-Lagrangian advection of scalars: phi(x,t+dt) = phi(x - u*dt, t)

DX = params['dx']; DYc = params['dy']; DG = 1.0  # Dg scaled to 1 in eqs7 nondimensionalization

# Precompute diffusion decays per dt
GammaG = np.exp(-DG * K2 * args.dt)
GammaX = np.exp(-DX * K2 * args.dt)
GammaY = np.exp(-DYc * K2 * args.dt)

# Helper: reaction RHS at a pointwise field state

def reaction_rhs(pG, pX, pY, forcing):
    Xtot = pX + X0
    Ytot = pY + Y0
    nonlinear_s  = s_par * (Xtot**3 - X0**3)
    nonlinear_xy = (Xtot**2 * Ytot - X0**2 * Y0)
    dG = - q_par * pG + g_par * pX
    dX = p_par * pG - (1.0 + b) * pX + u_cross * pY - nonlinear_s + nonlinear_xy + forcing
    dY = b * pX - u_cross * pY + (-nonlinear_xy + nonlinear_s)
    return dG, dX, dY

# Velocity RHS (compressible NS): du/dt = - (u·∇)u - c_s^2 ∇ ln ρ + ν (∇^2 u + (1/3)∇(∇·u))
# ρ = ρ0 + α_G(G0+pG) + α_X(X0+pX) + α_Y(Y0+pY)

def velocity_rhs(ux, uy, pG, pX, pY):
    G = G0 + pG; X = X0 + pX; Yf = Y0 + pY
    rho = rho0 + alphaG*G + alphaX*X + alphaY*Yf
    # grad ln rho
    rx, ry = grad(np.log(rho + 1e-12))
    # convective term
    ux_x, ux_y = grad(ux)
    uy_x, uy_y = grad(uy)
    convx = ux*ux_x + uy*ux_y
    convy = ux*uy_x + uy*uy_y
    # viscosity terms
    lap_ux = laplacian(ux)
    lap_uy = laplacian(uy)
    divu = ux_x + uy_y
    divx, divy = grad(divu)
    visc_x = lap_ux + (1.0/3.0)*divx
    visc_y = lap_uy + (1.0/3.0)*divy
    # RHS
    dux = -convx - cs2*rx + nu*visc_x
    duy = -convy - cs2*ry + nu*visc_y
    return dux, duy

# Main integration segmented

def integrate_segment(t0, t1, pG, pX, pY, ux, uy, next_frame_idx, frames_done):
    t = t0
    frame_times_seg = frame_times[(frame_times > t0 + 1e-12) & (frame_times <= t1 + 1e-12)]
    idx_ft = 0
    while t < t1 - 1e-12:
        dt = min(args.dt, t1 - t)
        # 1) Reaction RK2
        forcing = chi_xy_t(t)
        dG1, dX1, dY1 = reaction_rhs(pG, pX, pY, forcing)
        pG_tmp = pG + dt*dG1; pX_tmp = pX + dt*dX1; pY_tmp = pY + dt*dY1
        forcing2 = chi_xy_t(t + dt)
        dG2, dX2, dY2 = reaction_rhs(pG_tmp, pX_tmp, pY_tmp, forcing2)
        pG = pG + 0.5*dt*(dG1 + dG2)
        pX = pX + 0.5*dt*(dX1 + dX2)
        pY = pY + 0.5*dt*(dY1 + dY2)
        # 2) Diffusion via spectral decay
        pG = irfft2(GammaG * rfft2(pG))
        pX = irfft2(GammaX * rfft2(pX))
        pY = irfft2(GammaY * rfft2(pY))
        # 3) Velocity RK2
        dux1, duy1 = velocity_rhs(ux, uy, pG, pX, pY)
        ux_tmp = ux + dt*dux1; uy_tmp = uy + dt*duy1
        dux2, duy2 = velocity_rhs(ux_tmp, uy_tmp, pG, pX, pY)
        ux = ux + 0.5*dt*(dux1 + dux2)
        uy = uy + 0.5*dt*(duy1 + duy2)
        # optional mild viscosity filter (anti‑alias): diffuse velocity a little more
        filt = np.exp(-0.1 * K2 * dt)
        ux = irfft2(filt * rfft2(ux))
        uy = irfft2(filt * rfft2(uy))
        # 4) Advect scalars semi‑Lagrangian
        pG = advect_scalar(pG, ux, uy, dt)
        pX = advect_scalar(pX, ux, uy, dt)
        pY = advect_scalar(pY, ux, uy, dt)
        t += dt
        # render any frames that fall in this step
        while idx_ft < len(frame_times_seg) and frame_times_seg[idx_ft] <= t + 1e-12:
            tf = frame_times_seg[idx_ft]
            fidx = np.searchsorted(frame_times, tf)
            if fidx not in frames_done:
                render_surface(pG, pX, pY, ux, uy, tf, os.path.join(frames_dir, f'frame_{fidx:04d}.png'), zlim=args.zlim)
                frames_done.add(fidx)
                save_ckpt(t, pG, pX, pY, ux, uy, fidx+1, frames_done)
                live.maybe_update(fidx, pG, pX, pY, ux, uy, tf)
            idx_ft += 1
    return t, pG, pX, pY, ux, uy, next_frame_idx, frames_done

# ---------------- Main ----------------

def main():
    # Resume or fresh
    ck = load_ckpt()
    if ck is None:
        t_curr = 0.0
        next_frame_idx = 0
        frames_done = set()
    else:
        t_curr = ck['t_curr']
        global pG, pX, pY, ux, uy
        pG, pX, pY, ux, uy = ck['pG'], ck['pX'], ck['pY'], ck['ux'], ck['uy']
        next_frame_idx = ck['next_frame_idx']
        frames_done = ck['frames_done']
        print(f"[Resume] t={t_curr:.3f}, frames_done={len(frames_done)}")

    # Pre-render any frames at t=0
    if next_frame_idx < nt_anim and frame_times[next_frame_idx] <= t_curr + 1e-12:
        render_surface(pG, pX, pY, ux, uy, t_curr, os.path.join(frames_dir, f'frame_{next_frame_idx:04d}.png'), zlim=args.zlim)
        frames_done.add(next_frame_idx)
        next_frame_idx += 1
        save_ckpt(t_curr, pG, pX, pY, ux, uy, next_frame_idx, frames_done)

    t_start = time.time()

    while t_curr < Tfinal - 1e-12 and len(frames_done) < min(nt_anim, args.max_frames) and not live.quit:
        t_seg_end = min(Tfinal, t_curr + segment_dt)
        print(f"[Integrate] {t_curr:.3f} -> {t_seg_end:.3f}  (dt={dt})")
        t_curr, pG2, pX2, pY2, ux2, uy2, next_frame_idx, frames_done = integrate_segment(
            t_curr, t_seg_end, pG, pX, pY, ux, uy, next_frame_idx, frames_done)
        pG, pX, pY, ux, uy = pG2, pX2, pY2, ux2, uy2
        render_surface(pG, pX, pY, ux, uy, t_curr, final_png, zlim=args.zlim)
        save_ckpt(t_curr, pG, pX, pY, ux, uy, next_frame_idx, frames_done)
        print(f"  -> t={t_curr:.3f}/{Tfinal}, frames={len(frames_done)}/{nt_anim}, wall={time.time()-t_start:.1f}s")

    # Assemble MP4
    print('[Video] Writing MP4:', mp4_path)
    with imageio.get_writer(mp4_path, fps=max(8, int(nt_anim / max(1, Tfinal/2)))) as w:
        for i in range(min(nt_anim, args.max_frames)):
            f = os.path.join(frames_dir, f'frame_{i:04d}.png')
            if os.path.exists(f):
                img = imageio.imread(f)
                w.append_data(img)
    print('[Done] MP4 saved.')

if __name__ == '__main__':
    main()
