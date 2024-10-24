import matplotlib.pyplot
import pywt.data

# Load image
original = pywt.data.camera()

# Wavelet transform of image, and plot approximation and details
titles = ["Approximation", " Horizontal detail", "Vertical detail", "Diagonal detail"]
coeffs2 = pywt.dwt2(original, "bior1.3")
LL, (LH, HL, HH) = coeffs2
fig = matplotlib.pyplot.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=matplotlib.pyplot.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
matplotlib.pyplot.show()
