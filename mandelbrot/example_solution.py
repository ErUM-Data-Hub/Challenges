#!/usr/bin/env -S submit -f -M 2000 python -u

import numpy as np
import numba as nb
import numba.cuda as cu
import numba.cuda.random as ra
from time import time


np.set_printoptions(linewidth=300, suppress=True, precision=4)

W = 5000 # no. of workers (TUNEABLE)
L = 15000 # no. of loops (TUNEABLE)

# no. of patches per axix
Gb = 10  # (TUNEABLE)
Gv = 1 << Gb
Gm = Gv - 1

Gv2 = Gv * Gv # toal no. of patches

# no. of mantissa bits in float32
Fb = 24
Fv = 1 << Fb
Fm = Fv - 1
Fd = 1.0 / Fv

Er = 4.0 # escape radius^2

# patch size
Ex = 3 / Gv
Ey = 3 / Gv

# shorthands for common types
# numba.cuda often needs explicit casting
F = nb.float32
U = nb.uint32
B = nb.uint64
I = nb.int32


if True:
    dataD = cu.to_device(np.zeros((Gv, Gv, 2), dtype=np.int32))

    @cu.jit(inline=True)
    def add_count(data, x, y, num, esc):
        # only the first two threads in a warp get to do a atomic add, for
        # num(ber of sabmples) and (number of) esc(apes), reepectively
        if cu.laneid < 2:
            cu.atomic.add(
                data,
                (x, y, cu.laneid),
                cu.selp(cu.laneid == 0, num, esc),
            )

        cu.syncwarp()

else:
    # alternatively use uint64, but there is no atomic.add so
    # would need to use atomic.cas 


@cu.jit(inline=True)
def get_std(data, x, y):
    # uses Beta(pass + 1, fail + 1), should be extra conservative, see:
    # https://en.wikipedia.org/wiki/Beta_distribution#Bayes%E2%80%93Laplace_prior_probability_(Beta(1,1))
    a = data[x, y, 0] + 1
    b = data[x, y, 0] + 1

    n = a + b
    return F(a * b) / (F(n) * F(n) * F(n + 1))


@cu.jit(inline=True)
def warp_sum(val):
    val += cu.shfl_down_sync(0xFFFFFFFF, val, 1)
    val += cu.shfl_down_sync(0xFFFFFFFF, val, 2)
    val += cu.shfl_down_sync(0xFFFFFFFF, val, 4)
    val += cu.shfl_down_sync(0xFFFFFFFF, val, 8)
    val += cu.shfl_down_sync(0xFFFFFFFF, val, 16)

    return cu.shfl_up_sync(0xFFFFFFFF, val, cu.laneid)


@cu.jit
def calc(state, data, thresh: nb.float32):
    ti = I(cu.threadIdx.x)
    bi = I(cu.blockIdx.x)
    bn = I(cu.gridDim.x)
    gi = cu.grid(1)

    i = I(bi)
    more = False

    for _ in range(L):
        xi = i & Gm
        yi = (i >> Gb) & Gm

        x0 = F(xi) * F(Ex) - F(2.0)
        y0 = F(yi) * F(Ey) - F(1.5)

        while get_std(data, xi, yi) > thresh:
            more = True

            num = I(0)
            esc = I(0)
            res = I(-1)

            x = F(0)
            y = F(0)
            xh = F(0)
            yh = F(0)
            xt = F(0)
            yt = F(0)

            while cu.syncthreads_and(I(num < 25)):  # (TUNEABLE)

                num += cu.selp(res > 0, I(1), I(0))
                esc += cu.selp(res > 1, I(1), I(0))

                rdy = res == 0

                if True:
                    rand = ra.xoroshiro128p_next(state, gi)
                    rx = F(rand & U(Fm)) * F(Fd)
                    rand = U(rand >> B(Fb))
                    ry = F(rand & U(Fm)) * F(Fd)
                else:
                    rx = ra.xoroshiro128p_uniform_float32(state, gi)
                    ry = ra.xoroshiro128p_uniform_float32(state, gi)

                x = cu.selp(rdy, x, rx * F(Ex) + x0)
                y = cu.selp(rdy, y, ry * F(Ey) + y0)

                xh = cu.selp(rdy, F(xh), F(0))
                yh = cu.selp(rdy, F(yh), F(0))
                xt = cu.selp(rdy, F(xt), F(0))
                yt = cu.selp(rdy, F(yt), F(0))

                res = cu.selp(rdy, res, I(0))

                waste = I(0)
                while waste < 500:  # (TUNEABLE)
                    for _ in range(2):  # (TUNEABLE)
                        xt, yt = xt * xt - yt * yt + x, 2 * xt * yt + y
                        xh, yh = xh * xh - yh * yh + x, 2 * xh * yh + y
                        xh, yh = xh * xh - yh * yh + x, 2 * xh * yh + y

                        res = cu.selp(xh == xt and yh == yt, I(1), res)
                        res = cu.selp(xh * xh + yh * yh > F(Er), I(2), res)

                    waste += cu.syncthreads_count(I(res > 0))

            # sum up results in warp
            num = warp_sum(num)
            esc = warp_sum(esc)

            add_count(data, xi, yi, num, esc)

        # next position & wrap logic
        i += bn
        if i >= Gv2:
            i -= Gv2
            if i > Gv2:
                i = i % Gv2

            if not more:
                return
            more = False


def improve(thresh: float = 1e-3, seed=548932):
    state = ra.create_xoroshiro128p_states(W * 32, seed=seed)
    calc[W, 32](state, dataD, np.float32(thresh))
    return dataD.copy_to_host()


def D(val):
    return val.astype(np.float64)


if __name__ == "__main__":
    assert Gv2 < W * L, "some tiles would stay empty"

    print(
        "starting ...",
        f"tiles   : {Gv2:12} = {Gv2:8.2e}",
        f"blocks  : {W:12} = {W:8.2e}",
        f"loops   : {L:12} = {L:8.2e}",
        sep="\n",
    )
    dur = time()
    res = improve()
    dur = time() - dur

    res = res.astype(np.uint64)
    np.save("data.npy", res)

    tot = res[..., 0].sum()
    assert tot.all(), "some tiles are empty :("
    per = tot / Gv2
    print(
        f"samples : {tot:12} = {tot:8.2e}",
        f"per tile: {per:12.3f}",
        f"runtime : {dur:12.3f} s",
        f"rate    : {tot/dur:12.3e} samples/s",
        sep="\n",
    )

    num = res[..., 0]
    esc = res[..., 1]
    cap = num - esc

    area = Ex * Ey

    val = D(cap + 1) / D(num + 2)
    std = D(cap + 1) * D(esc + 1) / (D(num + 2) * D(num + 2) * D(num + 3))

    if False: # for debugging
        for name, value in dict(
            num=num,
            val=val,
            std=std,
            scaled_val=val * area,
            scaled_std=std * area,
        ).items():
            print(name)
            for prop in ["min", "max", "sum", "mean"]:
                v = getattr(value, prop)()
                if v.dtype == np.float64:
                    print(f"  {prop:4}: {v:20.8} = {v:9.3e}")
                else:
                    print(f"  {prop:4}: {v:20} = {v:9.3e}")

    from utils import confidence_interval, combine_uncertaintes, plot_pixels

    fig, ax, p = plot_pixels(val.T)
    fig.colorbar(p, ax=ax, shrink=0.75)
    fig.savefig("val.png")

    fig, ax, p = plot_pixels(std.T)
    fig.colorbar(p, ax=ax, shrink=0.75)
    fig.savefig("std.png")

    numer = cap
    denom = num

    CONFIDENCE_LEVEL = 0.05
    # CONFIDENCE_LEVEL /= 2  # two sided?
    # CONFIDENCE_LEVEL **= Gv2  # for the combination of tiles?

    val = np.sum(D(numer) / D(denom)) * area
    low, high = confidence_interval(CONFIDENCE_LEVEL, numer, denom, area)
    unc = combine_uncertaintes(low, high, denom)
    rel = unc / val

    print(
        "results :",
        f"val = {val:14.9f} = {val:8.2e}",
        f"unc = {unc:14.9f} = {unc:8.2e}",
        f"rel = {rel:14.9f} = {rel:8.2e}",
        sep="\n  ",
    )
