const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const http=require('node:http');
const {chromium,webkit}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const gameRoot=process.argv[2];
(async()=>{
 const server=http.createServer((req,res)=>{
  let file;
  if(req.url==='/audio.js'){file=path.join(__dirname,'../../matrix_game/web/audio.js');res.setHeader('Content-Type','text/javascript');}
  else if(/^\/audio\/[0-6]\.wav$/.test(req.url)){file=path.join(gameRoot,'assets/combat-audio',path.basename(req.url));res.setHeader('Content-Type','audio/wav');}
  else if(req.url==='/'){
   res.setHeader('Content-Type','text/html');res.end('<script src="/audio.js"></script><button id="start">Start</button><script>window.sound=new CastleAudio();document.querySelector("button").onclick=()=>sound.unlock().then(()=>window.ready=true).catch(e=>window.error=e.message);</script>');return;
  }else{res.writeHead(404);res.end();return;}
  res.end(fs.readFileSync(file));
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 try{
  for(const engine of ['chrome','webkit']){
   const browser=await (engine==='chrome'?chromium:webkit).launch({headless:true,...(engine==='chrome'&&process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
   try{
    const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);await page.click('#start');
    await page.waitForFunction(()=>window.ready||window.error);assert.equal(await page.evaluate(()=>window.error),undefined);
    const result=await page.evaluate(async()=>{
     const analyser=sound.context.createAnalyser();sound.mix.connect(analyser);analyser.fftSize=512;
     let peak=0;const timer=setInterval(()=>{const data=new Float32Array(512);analyser.getFloatTimeDomainData(data);for(const v of data)peak=Math.max(peak,Math.abs(v));},5);
     const durations=sound.buffers.map(b=>b.duration);
     for(let i=0;i<7;i++){
      const game={frame:i,simulation_time:i*.05,sounds:[{sound:i,gain:.8,pan:(i-3)/4,offset:-.01}]};
      sound.present([{game}],game,performance.now());sound.present([{game}],game,performance.now());
      await new Promise(r=>setTimeout(r,50));
     }
     const before=sound.status();sound.setEnabled(false);const muted=sound.status();sound.reset();
     clearInterval(timer);return {before,muted,after:sound.status(),durations,peak};
    });
    assert.deepEqual(result.before.bySound,Array(7).fill(1));assert.equal(result.muted.active,0);assert.equal(result.after.played,0);
    assert(result.peak>.001,'Silent actual Web Audio graph');
    assert(result.durations.every(d=>d>=.07&&d<=.86));assert.deepEqual(errors,[]);
    console.log(engine,JSON.stringify(result));
   }finally{await browser.close();}
  }
  console.log('AUDIO_BROWSER_PASS');
 }finally{await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
