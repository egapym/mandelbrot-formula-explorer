from pathlib import Path
from subprocess import check_output
# Experimental benchmark modules only; the application never imports them.
# python3 test/buildBuddhabrotGpuPrototypes.py
# Then open test/buddhabrotGpuProfileBrowserTests.html with:
# ?candidate=./fixtures/gpu-prototypes/interior.mjs,./fixtures/gpu-prototypes/queued.mjs,./fixtures/gpu-prototypes/compact.mjs
# Remove test/fixtures/gpu-prototypes after the experiment.
root = Path(__file__).resolve().parents[1]
output = root / 'test' / 'fixtures' / 'gpu-prototypes'
output.mkdir(parents=True, exist_ok=False)
# Pin the measured baseline: the production renderer now uses general queues.
base = check_output(['git', 'show', '4396f85:buddhabrotWebGPU.mjs'], cwd=root, text=True)
base = base.replace("from './", "from '../../../").replace("import('./", "import('../../../")
(output / 'before.mjs').write_text(base)
interior = base.replace('    const historyDeclarations = [', r"""    const skipInterior = (params.mode || 'buddha') === 'buddha' &&
      (params.fractalType || 'mandelbrot') === 'mandelbrot' &&
      (!params.iterationFunction || /^z\*z\+c$/.test(params.iterationFunction.replace(/\s/g, ''))) &&
      (params.z0Real || 0) === 0 && (params.z0Imag || 0) === 0 &&
      (params.escapeRadius ?? 4) >= 2
    const interiorRejection = skipInterior ? `
      let offset = c.x - 0.25;
      let q = offset * offset + c.y * c.y;
      let cardioid = q * (q + offset) - 0.25 * c.y * c.y;
      let bulb = (c.x + 1.0) * (c.x + 1.0) + c.y * c.y;
      if (cardioid < -0.00001 || bulb < 0.06249) { continue; }
    ` : ''
    const historyDeclarations = [""", 1)
interior = interior.replace('    var escaped = false;', '${interiorRejection}\n    var escaped = false;', 1)
(output / 'interior.mjs').write_text(interior)
source = interior.replace('    let samplesRemaining = totalSamples',
    '    let pendingSamples = 0\n    let queuedBatches = 0\n    let samplesRemaining = totalSamples', 1)
position = source.index('        // 次のバッチへ進む前に完了待ちする。')
source = source[:position] + """        pendingSamples += samplesThisBatch
        queuedBatches++
        if (!params.interactive && queuedBatches < 8 && samplesRemaining > samplesThisBatch) {
          samplesRemaining -= samplesThisBatch
          continue
        }

""" + source[position:]
source = source.replace('this.onProgress({ delta: samplesThisBatch, total: totalSamples })',
    'this.onProgress({ delta: pendingSamples, total: totalSamples })', 1)
source = source.replace('this.onProgress({ delta: processed, total: totalSamples })',
    'this.onProgress({ delta: pendingSamples, total: totalSamples })\n        pendingSamples = 0\n        queuedBatches = 0', 1)
(output / 'queued.mjs').write_text(source)

marker='    const module = device.createShaderModule({ code: wgsl })'
insertion=r'''
    let compactBuffers, classifyCode
    if (skipInterior && len <= 65535 * 64) {
      const declarations = `
@group(0) @binding(6) var<storage, read_write> sampleIds: array<u32>;
@group(0) @binding(7) var<storage, read_write> sampleCounts: array<atomic<u32>>;
`
      const entryStart = wgsl.indexOf('@compute @workgroup_size(64)\nfn main')
      const classifyStart = wgsl.slice(entryStart, wgsl.indexOf('var escaped = false;', entryStart))
        .replace('fn main(', 'fn classify(')
      classifyCode = wgsl.slice(0, entryStart) + declarations + classifyStart + `
      var escaped = false;
      var iter = 0u;
      loop {
        if (iter >= min(32u, iterationLimit)) { break; }
        let nextZ = ${customWGSLIteration};
        if (!(dot(nextZ, nextZ) <= escapeRadiusSquared)) { escaped = true; break; }
        z = nextZ; iter = iter + 1u;
      }
      if (escaped) {
        let slot = atomicAdd(&sampleCounts[0], 1u);
        sampleIds[slot] = id;
      } else if (iter < iterationLimit) {
        let slot = atomicAdd(&sampleCounts[1], 1u);
        sampleIds[u.width * u.height + slot] = id;
      }
    }
}
`
      wgsl = wgsl.slice(0, entryStart) + declarations + 'override queueKind: u32 = 0u;\n' + wgsl.slice(entryStart)
      wgsl = wgsl.replace('  let id = gid.x + u.invocationOffset;\n  if (id >= u.invocations) { return; }', `
  let slot = gid.x + u.invocationOffset;
  if (slot >= atomicLoad(&sampleCounts[queueKind])) { return; }
  let id = sampleIds[slot + queueKind * u.width * u.height];`)
      compactBuffers = {
        ids: device.createBuffer({ size: len * 8, usage: GPUBufferUsage.STORAGE }),
        counts: device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      }
    }
'''
assert marker in source
source=source.replace(marker,insertion+'\n'+marker,1)
# Raw strings above contain escaped newlines for the JS string literals only.
source=source.replace("'@compute @workgroup_size(64)\\\\nfn main'", "'@compute @workgroup_size(64)\\nfn main'")
source=source.replace("'override queueKind: u32 = 0u;\\\\n'", "'override queueKind: u32 = 0u;\\n'")
source=source.replace("'  let id = gid.x + u.invocationOffset;\\\\n  if", "'  let id = gid.x + u.invocationOffset;\\n  if")
source=source.replace("        { binding: 5, resource: { buffer: bandContribBuf } },", "        { binding: 5, resource: { buffer: bandContribBuf } },\n        ...(compactBuffers ? [\n          { binding: 6, resource: { buffer: compactBuffers.ids } },\n          { binding: 7, resource: { buffer: compactBuffers.counts } },\n        ] : []),",1)
marker='    const fractalType = params.fractalType || params.type || \'mandelbrot\''
# Insert extra pipelines after the main bind group, before uniform initialization.
marker='    const totalSamples ='
# Actual totalSamples precedes pipeline in this source, so use the bind group tail.
start=source.index('    const bindGroup = device.createBindGroup({')
end=source.index('\n    })',start)+len('\n    })')
source=source[:end]+r'''
    let classifyPipeline, classifyGroup, slowPipeline, slowGroup
    if (compactBuffers) {
      classifyPipeline = device.createComputePipeline({ layout: 'auto',
        compute: { module: device.createShaderModule({ code: classifyCode }), entryPoint: 'classify' } })
      classifyGroup = device.createBindGroup({ layout: classifyPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 6, resource: { buffer: compactBuffers.ids } },
        { binding: 7, resource: { buffer: compactBuffers.counts } },
      ] })
      slowPipeline = device.createComputePipeline({ layout: 'auto',
        compute: { module, entryPoint: 'main', constants: { queueKind: 1 } } })
      slowGroup = device.createBindGroup({ layout: slowPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: rBuf } },
        { binding: 2, resource: { buffer: gBuf } },
        { binding: 3, resource: { buffer: bBuf } },
        { binding: 4, resource: { buffer: bandBuf } },
        { binding: 5, resource: { buffer: bandContribBuf } },
        { binding: 6, resource: { buffer: compactBuffers.ids } },
        { binding: 7, resource: { buffer: compactBuffers.counts } },
      ] })
    }
''' +source[end:]
a=source.index('          const encoder = device.createCommandEncoder()', source.index('let samplesRemaining ='))
b=source.index('          device.queue.submit([encoder.finish()])',a)+len('          device.queue.submit([encoder.finish()])')
source=source[:a]+'''          const encoder = device.createCommandEncoder()
          const offView = new Uint32Array([invocationOffset])
          device.queue.writeBuffer(uniformBuf, 4 * 7, offView.buffer, 0, 4)
          if (compactBuffers) encoder.clearBuffer(compactBuffers.counts)
          for (const [p, group] of compactBuffers
            ? [[classifyPipeline, classifyGroup], [pipeline, bindGroup], [slowPipeline, slowGroup]]
            : [[pipeline, bindGroup]]) {
            const pass = encoder.beginComputePass()
            pass.setPipeline(p)
            pass.setBindGroup(0, group)
            pass.dispatchWorkgroups(thisChunk)
            pass.end()
          }
          device.queue.submit([encoder.finish()])'''+source[b:]
source=source.replace('    await readDensity()\n\n    return', '    await readDensity()\n    compactBuffers?.ids.destroy()\n    compactBuffers?.counts.destroy()\n\n    return',1)
p=output / 'compact.mjs'
p.write_text(source)
(output / 'compact-frame.mjs').write_text(source.replace(
    'const displayInterval = params.interactive ? 0 : 1000',
    'const displayInterval = params.interactive ? 16 : 1000'))

print(output)
