import os
import numpy as np
import matplotlib.pyplot as plt

# Para GIF
import imageio.v2 as imageio

np.random.seed(7)

ASSETS_DIR = os.path.join("../slides", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

def make_blobs(n=240, centers=3, std=0.75):
    C = np.array([[-2.5, -1.0],
                  [ 0.0,  2.5],
                  [ 2.5, -1.5]])[:centers]
    X = []
    per = n // centers
    for k in range(centers):
        X.append(C[k] + std*np.random.randn(per, 2))
    return np.vstack(X)

def kmeans_history(X, K=3, n_iters=8):
    n, _ = X.shape
    idx = np.random.choice(n, K, replace=False)
    centroids = X[idx].copy()

    history = []
    for it in range(n_iters):
        dists = ((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2)
        labels = np.argmin(dists, axis=1)
        history.append((centroids.copy(), labels.copy()))

        new_centroids = centroids.copy()
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    dists = ((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2)
    labels = np.argmin(dists, axis=1)
    history.append((centroids.copy(), labels.copy()))
    return history

def plot_scatter_raw(X, path):
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], s=18)
    plt.title("Datos sin etiquetas: ¿hay grupos naturales?")
    plt.xlabel("Característica 1"); plt.ylabel("Característica 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_state(X, centroids, labels, path, title):
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], s=18, c=labels)
    plt.scatter(centroids[:,0], centroids[:,1], s=180, marker="X")
    plt.title(title)
    plt.xlabel("Característica 1"); plt.ylabel("Característica 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def make_gif_from_history(X, history, gif_path):
    frames = []
    tmp_dir = os.path.join(ASSETS_DIR, "_frames_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    for i, (C, y) in enumerate(history):
        frame_path = os.path.join(tmp_dir, f"frame_{i:02d}.png")
        plot_state(X, C, y, frame_path, title=f"k-means — iteración {i}")
        frames.append(imageio.imread(frame_path))

    imageio.mimsave(gif_path, frames, duration=0.9)  # 0.9s por frame

def make_elbow_plot(X, path, Kmax=8):
    # Inercia (SSE) para ilustrar la idea de elegir K (sin entrar en detalle)
    def fit_kmeans_sse(X, K, n_iters=20):
        hist = kmeans_history(X, K=K, n_iters=n_iters)
        C, y = hist[-1]
        sse = 0.0
        for k in range(K):
            pts = X[y == k]
            if len(pts) > 0:
                sse += ((pts - C[k])**2).sum()
        return sse

    Ks = np.arange(1, Kmax+1)
    sses = [fit_kmeans_sse(X, int(K)) for K in Ks]

    plt.figure(figsize=(6,4))
    plt.plot(Ks, sses, marker="o")
    plt.title("Elegir K: idea del 'codo' (elbow)")
    plt.xlabel("K (número de grupos)")
    plt.ylabel("Suma de errores intra-grupo (SSE)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def make_color_quantization_demo(path):
    # Imagen sintética tipo “degradado”
    h, w = 120, 180
    img = np.zeros((h, w, 3), dtype=float)
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(0, 1, w)[None, :]
    img[..., 0] = xx
    img[..., 1] = yy
    img[..., 2] = 0.35 + 0.35*np.sin(2*np.pi*xx)

    Xrgb = img.reshape(-1, 3)
    K = 8
    hist = kmeans_history(Xrgb, K=K, n_iters=25)
    C, labels = hist[-1]
    img_q = C[labels].reshape(h, w, 3)

    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img_q)
    plt.title(f"Cuantizada con k-means (K={K})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def main():
    X = make_blobs(n=240, centers=3, std=0.75)

    # 1) Scatter raw
    plot_scatter_raw(X, os.path.join(ASSETS_DIR, "kmeans_scatter.png"))

    # 2) Historia + GIF
    hist = kmeans_history(X, K=3, n_iters=8)
    make_gif_from_history(X, hist, os.path.join(ASSETS_DIR, "kmeans_iter.gif"))

    # 3) Elbow
    make_elbow_plot(X, os.path.join(ASSETS_DIR, "kmeans_elbow.png"), Kmax=8)

    # 4) Aplicación visual: cuantización
    make_color_quantization_demo(os.path.join(ASSETS_DIR, "kmeans_quantization.png"))

    print(f"Listo. Assets generados en: {ASSETS_DIR}")

if __name__ == "__main__":
    main()
