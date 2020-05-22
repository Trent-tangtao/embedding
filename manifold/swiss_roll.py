from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.datasets import make_s_curve,make_swiss_roll
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)

ax = plt.subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t,)
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
plt.show()

print(t.shape)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
isomap = Isomap(n_components=2)

X_reduced_isomap = isomap.fit_transform(X)
X_reduced = lle.fit_transform(X)
# plt.title("Unrolled swiss roll using LLE", fontsize=14)
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, )
plt.scatter(X_reduced_isomap[:, 0], X_reduced_isomap[:, 1], c=t,)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.grid(True)

# save_fig("lle_unrolling_plot")
plt.show()
