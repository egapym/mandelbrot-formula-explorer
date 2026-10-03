// Frozen pre-optimization shader for GPU output parity tests.
// Based on Bert Baron's perturbation renderer; keep independent of production changes.
export function legacyPerturbationShader(workgroupSize, smooth, bailout, supersampling) {
    let smoothCode = ''
    if (smooth)
      smoothCode = `
            var nu = log2(log2(zzq)) - 1;
            var modf = modf(nu);
            iter = iter - i32(modf.whole);
            smoothBuffer[i] = u32(255.0 * (1.0 - modf.fract));
        `

    // Perturbation formula for Mandelbrot — also tracks lastZz for sign computation
    const perturbationCode = `
                    // Mandelbrot perturbation
                    let zz = z + ez * eExpFactor;
                    lastZz = zz;
                    zzq = dot(zz, zz);
                    if (zzq < zqErrorBound) {
                        ${supersampling > 0 ? 'allConverged = false; break;' : 'return;'}
                    }

                    let ez_2z = z + zz;
                    ez = vec2f(dot(ez_2z, vec2f(ez.x, -ez.y)), dot(ez_2z, vec2f(ez.y, ez.x))) + dc;
                `

    // Generate compute function based on supersampling mode
    let computeFunction = ''
    if (supersampling > 0) {
      const samples = supersampling
      computeFunction = `
            @compute @workgroup_size(${workgroupSize}) fn computeSomething(
              @builtin(global_invocation_id) id: vec3u
            ) {
                let iid = id.x;
                if (iid >= spec.size) {
                    return;
                }
                let i = u32(indexBuffer[iid]);  // input will always be >=0
                let xy = vec2f(f32(i % spec.w), f32(i / spec.w));

                // Supersampling: ${samples}x${samples} samples per pixel
                var totalIter = 0.0;
                var totalSmooth = 0.0;
                var sampleCount = 0;
                var allConverged = true;
                var capturedSign = 0u;
                var capturedZr = 0.0;
                var capturedZi = 0.0;

                for (var sy = 0; sy < ${samples}; sy++) {
                    for (var sx = 0; sx < ${samples}; sx++) {
                        let offset = vec2f((f32(sx) + 0.5) / ${samples}.0, (f32(sy) + 0.5) / ${samples}.0);
                        var dc = fma(xy + offset, spec.dd, spec.dd0) - spec.reff;

                        var eExp = spec.dExp;
                        var eExpFactor = spec.dExpFactor;
                        var ez = dc;
                        var lastZz = vec2f(0.0, 0.0);

                        var iter = -1;
                        var zzq = 0.0;
                        while (zzq <= ${bailout}) {
                            iter = iter + 1;
                            if (iter == spec.max_iter) {
                                totalIter += 2.0;
                                sampleCount++;
                                break;
                            }
                            if (iter >= spec.refSize) {
                                allConverged = false;
                                break;
                            }

                            while (max(abs(ez.x), abs(ez.y)) > 2) {
                                eExp = eExp + 1.0;
                                ez = ez * 0.5;
                                dc = dc * 0.5;
                                eExpFactor = eExpFactor * 2.0;
                                if (eExp == -126.0) {
                                    eExpFactor = 0x1.0p-126;
                                }
                            }

                            let z = zBuffer[iter];
                            let zqErrorBound = zqErrorBoundBuffer[iter];

                            ${perturbationCode}
                        }

                        if (iter >= 0 && iter < spec.max_iter && zzq > ${bailout}) {
                            // Capture sign from first sample (sy=0, sx=0)
                            if (sy == 0 && sx == 0) {
                                let sameSign = (lastZz.x >= 0.0) == (lastZz.y >= 0.0);
                                capturedSign = select(2u, 1u, sameSign);
                                capturedZr = lastZz.x;
                                capturedZi = lastZz.y;
                            }
                            ${
                              smooth
                                ? `
                            var nu = log2(log2(zzq)) - 1;
                            var modf_result = modf(nu);
                            let adjustedIter = f32(iter) - modf_result.whole;
                            let smoothValue = 255.0 * (1.0 - modf_result.fract);
                            totalIter += adjustedIter + 4.0;
                            totalSmooth += smoothValue;
                            `
                                : `
                            totalIter += f32(iter) + 4.0;
                            `
                            }
                            sampleCount++;
                        }
                    }
                }

                if (!allConverged || sampleCount == 0) {
                    return;
                }

                values[i] = i32(round(totalIter / f32(sampleCount)));
                ${smooth ? 'smoothBuffer[i] = u32(round(totalSmooth / f32(sampleCount)));' : 'smoothBuffer[i] = 0;'}
                signsBuffer[i] = capturedSign;
                zrealBuffer[i] = capturedZr;
                zimagBuffer[i] = capturedZi;
                indexBuffer[iid] = -1;
            }
            `
    } else {
      computeFunction = `
            @compute @workgroup_size(${workgroupSize}) fn computeSomething(
              @builtin(global_invocation_id) id: vec3u
            ) {
                let iid = id.x;
                if (iid >= spec.size) {
                    return;
                }
                let i = u32(indexBuffer[iid]);  // input will always be >=0
                let xy = vec2f(f32(i % spec.w), f32(i / spec.w));
                var dc = fma(xy, spec.dd, spec.dd0) - spec.reff;

                var eExp = spec.dExp;
                var eExpFactor = spec.dExpFactor;

                var ez = dc;
                var lastZz = vec2f(0.0, 0.0);

                var iter = -1;
                var zzq = 0.0;
                while (zzq <= ${bailout}) {
                    iter = iter + 1;
                    if (iter == spec.max_iter) {
                        values[i] = 2;
                        smoothBuffer[i] = 0;
                        indexBuffer[iid] = -1;
                        return;
                    }
                    if (iter >= spec.refSize) {
                        return;
                    }

                    while (max(abs(ez.x), abs(ez.y)) > 2) {
                        eExp = eExp + 1.0;
                        ez = ez * 0.5;
                        dc = dc * 0.5;
                        eExpFactor = eExpFactor * 2.0;
                        if (eExp == -126.0) {
                            eExpFactor = 0x1.0p-126;
                        }
                    }

                    let z = zBuffer[iter];
                    let zqErrorBound = zqErrorBoundBuffer[iter];

                    ${perturbationCode}
                }

                ${smoothCode}

                let sameSign = (lastZz.x >= 0.0) == (lastZz.y >= 0.0);
                signsBuffer[i] = select(2u, 1u, sameSign);
                zrealBuffer[i] = lastZz.x;
                zimagBuffer[i] = lastZz.y;
                values[i] = iter + 4;
                indexBuffer[iid] = -1;
            }
            `
    }

    //language=WGSL
    return `
            struct Spec {
                max_iter: i32,
                size: u32,
                refSize: i32,
                w: u32,
                h: u32,
                padd0: u32,
                reff: vec2f,
                dd0: vec2f,
                dd: vec2f,
                dExp: f32,
                dExpFactor: f32,
            };
            @group(0) @binding(0) var<uniform> spec: Spec;
            @group(0) @binding(1) var<storage, read_write> indexBuffer: array<i32>;
            @group(0) @binding(2) var<storage, read_write> values: array<i32>;
            @group(0) @binding(3) var<storage, read> zBuffer: array<vec2f>;
            @group(0) @binding(4) var<storage, read> zqErrorBoundBuffer: array<f32>;
            @group(0) @binding(5) var<storage, read_write> smoothBuffer: array<u32>;
            @group(0) @binding(6) var<storage, read_write> signsBuffer: array<u32>;
            @group(0) @binding(7) var<storage, read_write> zrealBuffer: array<f32>;
            @group(0) @binding(8) var<storage, read_write> zimagBuffer: array<f32>;

            /**
             * This is the authors own code. In particular, the idea to use an implicit
             * extended exponent to overcome the limit of float32 is the authors own.
             * Feel free to use or adapt this code in your own projects.
             * If you do, I would greatly appreciate it if you could reference the original source.
             * Thank you!
             */

            ${computeFunction}
        `
  }
