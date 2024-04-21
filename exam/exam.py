import taichi as ti
ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=ti.f32, shape=(2*n, 2*n))


@ti.func
def paint_for_one_pixel(
    pixels: ti.template(),
    i: int,
    j: int
):
    if abs(i**2-j) <= 3:
        pixels[i, j] = 1
    else:
        pixels[i, j] = 0


@ti.kernel
def paint(pixels: ti.template()):
    for i, j in pixels:
        paint_for_one_pixel(pixels, i, j)


gui = ti.GUI("x^2 function", res=(2*n, 2*n))
while gui.running:
    paint(pixels)
    gui.set_image(pixels)
    gui.show()
