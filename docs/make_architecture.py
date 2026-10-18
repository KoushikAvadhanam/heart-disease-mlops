"""Generate docs/architecture.png and .svg (run: python docs/make_architecture.py)."""
import matplotlib
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")

LAYER = {"A": "#eaf2fb", "B": "#fdf3e3", "C": "#eaf7ec"}
BOX = {"A": "#4a90d9", "B": "#e69a2e", "C": "#3fa55b"}


def layer(y0, y1, key, title):
    ax.add_patch(FancyBboxPatch((1, y0), 94, y1 - y0, boxstyle="round,pad=0.2,rounding_size=1.5",
                                fc=LAYER[key], ec=BOX[key], lw=1.5))
    ax.text(2.2, y1 - 1.3, title, fontsize=12, fontweight="bold", color=BOX[key], va="top")


def box(x, y, w, h, key, text):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=1",
                                fc="white", ec=BOX[key], lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)
    return (x, y, w, h)


def arrow(p, q, label=None, dy=1.2, dx=0):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333333"))
    if label:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label, ha="center",
                fontsize=8, style="italic", color="#333333")


ax.text(50, 61, "Heart Disease Risk API - MLOps Architecture", ha="center",
        fontsize=16, fontweight="bold")

# Layer A: development
layer(40, 59.5, "A", "1. Data and model development")
w, h, y = 15, 9, 47
box(3, y, w, h, "A", "UCI Heart Disease\n(Cleveland, 303 rows)")
box(22, y, w, h, "A", "download_data.py\nclean + binary target\n-> data/heart.csv")
box(41, y, w, h, "A", "EDA\nhistograms, heatmap,\nclass balance")
box(60, y, w, h, "A", "train.py\nsklearn Pipeline\nLR + RF, GridSearchCV\n5-fold CV")
box(79, y, w, h, "A", "models/model.joblib\n(preprocessing +\nbest model)")
box(60, 41, w, 4, "A", "MLflow tracking")
for x in (18, 37, 56, 75):
    arrow((x + 0.3, y + h / 2), (x + 3.7, y + h / 2))
arrow((67.5, y), (67.5, 45), "params, metrics, plots", dy=0, dx=7)

# Layer B: CI/CD
layer(24.5, 38, "B", "2. CI/CD - GitHub Actions")
y, h = 25, 8
box(3, y, w, h, "B", "git push / PR\n(GitHub repo)")
box(22, y, w, h, "B", "Lint\nflake8")
box(41, y, w, h, "B", "Unit tests\npytest")
box(60, y, w, h, "B", "Train job\nuploads model +\nMLflow artifacts")
box(79, y, w, h, "B", "Docker build +\ncontainer smoke test")
for x in (18, 37, 56, 75):
    arrow((x + 0.3, y + h / 2), (x + 3.7, y + h / 2))

# Layer C: deployment + monitoring
layer(1.5, 23, "C", "3. Deployment and monitoring - Kubernetes (Minikube / Docker Desktop)")
box(79, 10.5, w, 8, "C", "Docker image\nheart-api:latest\n(FastAPI + uvicorn)")
box(60, 10.5, w, 8, "C", "Deployment\n2 pods, probes\n/predict /health /metrics")
box(41, 10.5, w, 8, "C", "Service\nLoadBalancer\n:80 -> :8000")
box(22, 10.5, w, 8, "C", "Client\ncurl / app\nPOST /predict")
box(60, 2.5, w, 5, "C", "Prometheus")
box(41, 2.5, w, 5, "C", "Grafana dashboard")
arrow((78.7, 14.5), (75.3, 14.5), "pull", dy=0.9)
arrow((56.3, 14.5), (59.7, 14.5))
arrow((37.3, 14.5), (40.7, 14.5))
arrow((59.7, 5), (56.3, 5), "query", dy=0.9)
arrow((67.5, 7.6), (67.5, 10.4), "scrape /metrics", dy=0, dx=6.5)

# Cross-layer arrows
arrow((86.5, 25), (86.5, 18.6), "build image", dy=-0.2, dx=6)
ax.plot([94, 97, 97], [51.5, 51.5, 14.5], color="#333333", lw=1.8)
arrow((97, 14.5), (94.3, 14.5))
ax.text(97.4, 33, "COPY model\ninto image", fontsize=8, style="italic",
        rotation=90, va="center")

fig.savefig("docs/architecture.png", dpi=200, bbox_inches="tight")
fig.savefig("docs/architecture.svg", bbox_inches="tight")
print("saved")
