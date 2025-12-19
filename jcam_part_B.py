import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Optional SSIM
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

# -------------------------
# Padding as extension operator
# -------------------------
def pad_image(u, mode="reflect"):
    """
    Extend u beyond its boundaries according to the chosen padding mode.
    mode: 'zero', 'reflect', 'replicate', 'wrap'
    """
    if mode == "zero":
        return np.pad(u, pad_width=1, mode='constant', constant_values=0)
    elif mode == "reflect":
        return np.pad(u, pad_width=1, mode='reflect')
    elif mode == "replicate":
        return np.pad(u, pad_width=1, mode='edge')
    elif mode == "wrap":
        return np.pad(u, pad_width=1, mode='wrap')
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

def unpad_image(u_padded):
    """Remove the 1-pixel padding."""
    return u_padded[1:-1, 1:-1]

def _scipy_mode_from_pad(mode):
    """Map padding keywords to scipy.ndimage.gaussian_filter modes."""
    mapping = {"zero": "constant", "reflect": "reflect", 
               "replicate": "nearest", "wrap": "wrap"}
    if mode not in mapping:
        raise ValueError(f"Unknown padding mode: {mode}")
    return mapping[mode]

# -------------------------
# PDE components
# -------------------------
def add_gaussian_noise(u, sigma=0.05):
    """Add Gaussian noise to image u."""
    noisy = u + np.random.randn(*u.shape) * sigma
    return np.clip(noisy, 0.0, 1.0)

def compute_laplacian(u, pad_mode):
    """Compute discrete Laplacian with padding."""
    u_pad = pad_image(u, mode=pad_mode)
    lap = (u_pad[2:, 1:-1] + u_pad[:-2, 1:-1] +
           u_pad[1:-1, 2:] + u_pad[1:-1, :-2] - 4 * u_pad[1:-1, 1:-1])
    return lap

def restore_channel(f, lam=1.0, tau=0.0001, max_iter=300, tol=1e-3, 
                    sigma=3.0, pad_mode="reflect"):
    """
    Restore a single channel using the PDE: -Laplacian(u) + lambda*G(u) = f
    Parameters: lam=1.0 (regularization), tau=0.0001 (time step), sigma=3.0 (Gaussian kernel)
    """
    u = f.copy()
    residuals = []
    for it in range(max_iter):
        lap = compute_laplacian(u, pad_mode)
        u_pad = pad_image(u, mode=pad_mode)
        gf_mode = _scipy_mode_from_pad(pad_mode)
        nonlocal_term = unpad_image(gaussian_filter(u_pad, sigma=sigma, mode=gf_mode))
        res = f - ((-lap) + lam * nonlocal_term)
        u += tau * res
        res_norm = np.linalg.norm(res.ravel())
        residuals.append(res_norm)
        if res_norm < tol:
            break
    return u, residuals

def restore_image_color(f, lam=1.0, tau=0.0001, max_iter=300, tol=1e-3, 
                        sigma=3.0, pad_mode="reflect"):
    """Restore color image channel-by-channel."""
    out = np.zeros_like(f)
    all_residuals = []
    for c in range(3):
        out[..., c], res = restore_channel(f[..., c], lam=lam, tau=tau,
                                           max_iter=max_iter, tol=tol, 
                                           sigma=sigma, pad_mode=pad_mode)
        all_residuals.append(res)
    return out, all_residuals

# -------------------------
# Metrics and postprocessing
# -------------------------
def compute_metrics(orig, proc):
    """Compute MSE, PSNR, and SSIM between original and processed images."""
    mse = np.mean((orig - proc)**2)
    psnr = float('inf') if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))
    s = None
    if ssim is not None:
        s = sum(ssim(orig[:,:,c], proc[:,:,c], data_range=1.0) for c in range(3)) / 3.0
    return mse, psnr, s

def _to_uint8(img_float):
    return np.clip(np.rint(img_float * 255.0), 0, 255).astype(np.uint8)

def apply_nlm(img):
    """Apply Non-Local Means denoising."""
    bgr = cv2.cvtColor(_to_uint8(img), cv2.COLOR_RGB2BGR)
    bgr_denoised = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=15, hColor=15, templateWindowSize=7, searchWindowSize=21)
    return cv2.cvtColor(bgr_denoised, cv2.COLOR_BGR2RGB) / 255.0

def apply_median(img, ksize=3):
    """Apply median filtering."""
    bgr = cv2.cvtColor(_to_uint8(img), cv2.COLOR_RGB2BGR)
    bgr_filtered = cv2.medianBlur(bgr, ksize=ksize)
    return cv2.cvtColor(bgr_filtered, cv2.COLOR_BGR2RGB) / 255.0

def apply_nlm_median(img):
    """Apply NLM followed by median filtering."""
    return apply_median(apply_nlm(img))

# -------------------------
# Main experiment: Corrected protocol (degrade first, then restore)
# -------------------------
def main():
    # 1. Load clean image
    bgr = cv2.imread("lenna.png", cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("lenna.png not found")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    clean = rgb.astype(np.float32) / 255.0

    # 2. PDE parameters (as discussed in the parameter selection section)
    lam, tau = 1.0, 0.0001  # lambda=1.0, tau=0.0001 satisfies stability condition
    max_iter, tol, sigma_G = 300, 1e-3, 3.0  # sigma_G=3.0 for Gaussian kernel
    pad_mode = "reflect"  # Neumann-like boundary conditions

    # 3. Test various noise levels - CORRECT PROTOCOL: degrade FIRST, then restore
    sigmas_noise = [0.09, 0.11, 0.15, 0.18]
    
    print("="*80)
    print("ABLATION STUDY: Evaluating contribution of each pipeline component")
    print("Protocol: Clean -> Add Noise -> Apply Restoration -> Compare to Clean")
    print("="*80)

    for sn in sigmas_noise:
        print(f"\n{'='*60}")
        print(f"Noise level sigma = {sn:.2f}")
        print('='*60)
        
        # Step 1: Degrade the clean image FIRST
        noisy = add_gaussian_noise(clean, sigma=sn)
        mse_noisy, psnr_noisy, ssim_noisy = compute_metrics(clean, noisy)
        print(f"Noisy input:        PSNR={psnr_noisy:.2f} dB, SSIM={ssim_noisy:.4f}")
        
        # Ablation Configuration 1: PDE only (no post-processing)
        pde_restored, residuals = restore_image_color(noisy, lam, tau, max_iter, 
                                                       tol, sigma_G, pad_mode)
        mse_pde, psnr_pde, ssim_pde = compute_metrics(clean, pde_restored)
        print(f"PDE only:           PSNR={psnr_pde:.2f} dB, SSIM={ssim_pde:.4f}")
        
        # Ablation Configuration 2: PDE + NLM
        pde_nlm = apply_nlm(pde_restored)
        mse_pde_nlm, psnr_pde_nlm, ssim_pde_nlm = compute_metrics(clean, pde_nlm)
        print(f"PDE + NLM:          PSNR={psnr_pde_nlm:.2f} dB, SSIM={ssim_pde_nlm:.4f}")
        
        # Ablation Configuration 3: PDE + NLM + Median (full pipeline)
        pde_nlm_med = apply_median(pde_nlm)
        mse_full, psnr_full, ssim_full = compute_metrics(clean, pde_nlm_med)
        print(f"PDE + NLM + Median: PSNR={psnr_full:.2f} dB, SSIM={ssim_full:.4f}")
        
        # Ablation Configuration 4: NLM + Median only (no PDE)
        nlm_med_only = apply_nlm_median(noisy)
        mse_nlm_med, psnr_nlm_med, ssim_nlm_med = compute_metrics(clean, nlm_med_only)
        print(f"NLM + Median only:  PSNR={psnr_nlm_med:.2f} dB, SSIM={ssim_nlm_med:.4f}")
        
        # PDE contribution = full pipeline - post-processing alone
        pde_contribution = psnr_full - psnr_nlm_med
        print(f"\nPDE contribution:   +{pde_contribution:.2f} dB over post-processing alone")

        # Visual comparison
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        images = [clean, noisy, pde_restored, pde_nlm, pde_nlm_med, nlm_med_only]
        titles = ["Clean Original", f"Noisy (sigma={sn})", "PDE Only", 
                  "PDE + NLM", "PDE + NLM + Median", "NLM + Median Only"]
        for ax, img, title in zip(axes.flat, images, titles):
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title)
            ax.axis('off')
        plt.suptitle(f"Ablation Study: sigma = {sn}")
        plt.tight_layout()
        plt.savefig(f"Figure_restoration_{sn:.2f}.png", dpi=150)
        plt.show()
        
        # Plot convergence (residual vs iteration)
        plt.figure(figsize=(8, 4))
        for c, color in enumerate(['r', 'g', 'b']):
            plt.semilogy(residuals[c], color=color, alpha=0.7, label=f'Channel {c}')
        plt.xlabel('Iteration')
        plt.ylabel('Residual norm (log scale)')
        plt.title(f'Convergence of Fixed-Point Iteration (sigma={sn})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Figure_convergence_{sn:.2f}.png", dpi=150)
        plt.show()

if __name__ == "__main__":
    main()