const result = document.getElementById('result')
try {
  const {CpuDensityColorMap} = await import('../buddhabrotCpuColorMap.mjs')
  const {colorMapDensity} = await import('../buddhabrotWebGPU.mjs')
  const results=[]
  for(const gamma of [0.1,4.8,20]) for(const factor of [1e-9,1,1e9]) for(const brightness of [0.05,1.2,6]) {
    const r=Float32Array.from({length:256},(_,i)=>factor*2**(i/20))
    const g=Float32Array.from(r,v=>v*.5), b=Float32Array.from(r,v=>v*.1)
    const cpu=new Uint8ClampedArray(256*4)
    new CpuDensityColorMap(r,g,b,256,1,cpu).update(brightness,gamma)
    const bitmap=await colorMapDensity({rBuf:r,gBuf:g,bBuf:b,width:256,height:1,brightness,gamma})
    const canvas=document.createElement('canvas');canvas.width=256;canvas.height=1
    const ctx=canvas.getContext('2d');ctx.drawImage(bitmap,0,0);bitmap.close()
    const gpu=ctx.getImageData(0,0,256,1).data
    const delta=Math.max(...cpu.map((v,i)=>Math.abs(v-gpu[i])))
    if(delta>1) throw Error(`CPU/GPU mismatch ${delta}`)
    results.push({gamma,factor,brightness,delta})
  }
  result.textContent = 'PASS: CPU/WebGPU exposure parity across gamma and density scales\n' + JSON.stringify(results, null, 2)
} catch (error) {
  result.textContent = 'FAIL: ' + error.stack
  throw error
}
