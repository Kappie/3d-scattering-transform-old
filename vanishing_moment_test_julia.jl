"""Computes `gauss{T<:Real}(ω, den::T) = exp(- ω*ω/den)`.
The Gaussian bell curve is defined as `gauss(ω) = exp(- ω² / 2σ²)`.
For performance reasons, we memoize the denominator `2σ²`,
which is computed only once in the caller `morlet`.
Also note that the exponentiation `ω²` is replaced by the explicit
product `ω*ω`.
"""
gauss(ω, den) = exp(-ω*ω/den)


function morlet(center::Int, den::AbstractFloat, N::Int, nPeriods::Int)
    """1. **Compute range of frequencies with nonneglible magnitude**"""
    halfN = N >> 1
    pstart = - ((nPeriods-1)>>1)
    pstop = (nPeriods-1)>>1 + iseven(nPeriods)
    ωstart = - halfN + pstart * N
    ωstop = halfN + pstop * N - 1
    """2. **Compute Gaussians**"""
    @inbounds begin
        gauss_center = [ gauss(ω-center, den) for ω in ωstart:ωstop ]
        gauss_0 = [ gauss(ω, den)
            for ω in (ωstart + pstart*N):(ωstop + pstop*N) ]
        corrective_gaussians = [ gauss_0[1 + ω + p*N]
            for ω in 0:(N*nPeriods-1), p in 0:(nPeriods-1) ]
    end
    """3. **Solve linear system to find out the corrective factors**"""
    b = [ gauss(p*N - center, den) for p in pstart:pstop ]
    A = [ gauss((q-p)*N, den)
        for p in 0:(nPeriods-1), q in 0:(nPeriods-1) ]
    corrective_factors = A \ b
    """4. **Periodize in Fourier domain**"""
    y = gauss_center - corrective_gaussians * corrective_factors
    y = reshape(y, N, nPeriods)
    y = squeeze(sum(y, 2), 2)
end

N = convert(Int, 32)
xi_radians = 2*pi/5
xi = convert(Int, ceil( (xi_radians/(2*pi))*N ))
sigma_spatial = 0.8
sigma_fourier = 1/sigma_spatial
den = 2*sigma_fourier*sigma_fourier
n_periods = 11

result = morlet(xi, den, N, n_periods)
display(result)
