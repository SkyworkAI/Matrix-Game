const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium, webkit} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const baseURL = process.env.TEST_URL || 'http://127.0.0.1:8025';

(async () => {
 const output = path.resolve(process.argv[2]);fs.mkdirSync(output,{recursive:true});
 const safari = process.env.TEST_WEBKIT === '1';
 const displayOnly = process.env.DISPLAY_ONLY === '1';
 const browser = await (safari ? webkit : chromium).launch({headless:true,...(!safari&&process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const page = await browser.newPage({viewport:{width:1280,height:800}});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 page.on('console',m=>{if(m.type()==='error')console.log('Browser console:',m.text());});
 const results=[];
 async function inspect(label){
  const value=await page.evaluate(()=>{
   const v=document.querySelector('video'),c=document.createElement('canvas');c.width=96;c.height=54;
   const ctx=c.getContext('2d');ctx.drawImage(v,0,0,c.width,c.height);const a=ctx.getImageData(0,0,c.width,c.height).data;
   let sum=0,sq=0;for(let i=0;i<a.length;i+=4){sum+=a[i];sq+=a[i]*a[i];}const n=a.length/4;
   const halves=[0,48].map(start=>{let s=0,sq=0;for(let y=0;y<54;y++)for(let x=start;x<start+48;x++){const p=a[(y*96+x)*4];s+=p;sq+=p*p;}return sq/(48*54)-(s/(48*54))**2;});
   const hud=document.querySelector('#hud');
   return {...window.castleDebug(),videoWidth:v.videoWidth,videoHeight:v.videoHeight,halves,hudVisible:!hud.hidden,
    hudPixels:Array.from(hud.getContext('2d').getImageData(0,0,hud.width,hud.height).data).filter((_,i)=>i%4===3).filter(v=>v>0).length,
    pixelMean:sum/n,pixelVariance:sq/n-(sum/n)**2};
  });
  assert(value.pixelVariance>50,`${label}: blank video`);
  assert(value.presented>10&&value.matched>0,`${label}: missing video/state`);
  assert.equal(value.videoWidth,value.monitor?1728:864);assert.equal(value.videoHeight,480);
  assert(value.hudVisible&&value.hudPixels>100,'Missing native-style HUD');
  assert.equal(value.hud.health,`HP ${Math.trunc(value.game.health)}`);
  assert.equal(value.hud.weapon,['PISTOL','RIFLE','SHOTGUN'][value.game.weapon]);
  const scene=process.env.TEST_SCENE||'castle';
  assert.equal(value.server.scene,scene);
  assert.equal(value.server.trees,scene==='castle'?4:608);
  for(const k of ['owned','raised','trap_cooldown','cover_status'])assert(k in value.game,`Missing ${k}`);
  if(value.monitor)assert(value.halves.every(v=>v>50),'Blank dual-view panel');
  results.push({label,...value});console.log(label,JSON.stringify(value));
  await page.screenshot({path:path.join(output,label+'.png')});return value;
 }
 async function run(mode,seconds,monitor=false){
  const label=mode+(monitor?'-dual':'');
  await page.selectOption('#mode',mode);
  if(mode==='model')await page.setChecked('#monitor',monitor);
  else assert(await page.locator('#monitor').isDisabled());
  await page.click('#play');
  await page.waitForFunction(()=>window.castleDebug().session&&window.castleDebug().gameFrame>10&&window.castleDebug().presented>12,{},{timeout:90000});
  await page.click('#surface');
  if(!displayOnly)await page.waitForFunction(()=>window.castleDebug().locked,{},{timeout:5000});
  await page.waitForFunction(()=>window.castleDebug().audio.ready&&window.castleDebug().audio.state==='running',{},{timeout:10000});
  await page.evaluate(()=>{
   const analyser=sound.context.createAnalyser();analyser.fftSize=256;sound.mix.connect(analyser);
   window.audioProbe={peak:0,samples:0};
   window.audioProbeTimer=setInterval(()=>{const values=new Float32Array(256);analyser.getFloatTimeDomainData(values);window.audioProbe.peak=Math.max(window.audioProbe.peak,...values.map(Math.abs));window.audioProbe.samples++;},10);
  });
  const initial=await inspect(label+'-initial');
  if(!displayOnly){
  const inputRate=await page.evaluate(async()=>{
   const socket=ws,original=socket.send.bind(socket);let packets=0;
   socket.send=value=>{const msg=JSON.parse(value);if(msg.look_x)packets++;original(value);};
   const start=performance.now();
   const timer=setInterval(()=>{const event=new MouseEvent('mousemove');Object.defineProperty(event,'movementX',{value:20});document.dispatchEvent(event);},4);
   try{await new Promise(resolve=>setTimeout(resolve,1200));}
   finally{clearInterval(timer);socket.send=original;resetInput();}
   return packets*1000/(performance.now()-start);
  });
  assert(inputRate>70&&inputRate<160,`Unexpected mouse packet rate ${inputRate}`);
  results.push({label:label+'-mouse',inputRate});
  assert.equal(initial.server.mouse_input_schema,2);
  assert.equal(initial.server.camera_max_degrees_per_second,90);
  async function fireAndHear(sound){
   const before=await page.evaluate(i=>window.castleDebug().audio.bySound[i],sound);
   for(let attempt=0;attempt<3;attempt++){
    await page.mouse.down();await page.waitForTimeout(350);await page.mouse.up();
    await page.waitForTimeout(2500);
    const after=await page.evaluate(i=>window.castleDebug().audio.bySound[i],sound);
    if(after>before)return;
   }
   assert((await page.evaluate(i=>window.castleDebug().audio.bySound[i],sound))>before,`Sound ${sound} was not heard`);
  }
  await page.waitForTimeout(2200); // Native spawn cooldown is up to two seconds.
  await fireAndHear(0);
  await page.keyboard.down('w');await page.waitForTimeout(1000);await page.keyboard.up('w');
  await page.mouse.move(700,400,{steps:10});
  await page.keyboard.press('2');
  await page.waitForFunction(()=>window.castleDebug().game?.weapon===1,{},{timeout:15000});
  await page.waitForTimeout(500); // HUD weapon changes before the equip cooldown expires.
  await fireAndHear(2);
  await page.waitForTimeout(4000);
  await page.keyboard.press('3');
  await page.waitForFunction(()=>window.castleDebug().game?.weapon===2,{},{timeout:15000});
  await page.waitForTimeout(500);
  await fireAndHear(1);
  }
  await page.waitForTimeout(seconds*1000);
  const end=await inspect(label+'-final');assert(end.gameFrame>initial.gameFrame+40);
  assert.equal(end.server.road_surface_splits,process.env.TEST_SCENE==='ancient-egypt'?3:0);
  assert.equal(end.server.fade_small_edges,false);
  if(!displayOnly)assert(end.audio.played>0&&end.audio.bySound.slice(0,4).every(n=>n>0),'Missing gunshot/explosion playback');
  const probe=await page.evaluate(()=>{clearInterval(window.audioProbeTimer);return window.audioProbe;});
  if(!displayOnly)assert(probe.peak>.0001,'Audio graph is silent');results.push({label:label+'-audio',...probe});
  if(!displayOnly)assert(end.game.camera.eye.some((v,i)=>Math.abs(v-initial.game.camera.eye[i])>.1),'Movement did not reach game');
  await page.evaluate(()=>document.exitPointerLock());
  if(monitor){
   for(const viewport of [{width:390,height:844},{width:1920,height:1080}]){
    await page.setViewportSize(viewport);await page.waitForTimeout(500);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    const bounds=await page.locator('#surface').boundingBox();assert(bounds.width<=viewport.width&&bounds.y+bounds.height<=viewport.height);
    await page.screenshot({path:path.join(output,`${label}-${viewport.width}.png`)});
   }
   await page.setViewportSize({width:1280,height:800});
  }
  const stopped=page.waitForResponse(r=>r.url().endsWith('/stop')&&r.request().method()==='POST');
  await page.click('#stop');
  assert([200,410].includes((await stopped).status()),'Stop request rejected');
  await page.waitForFunction(()=>!document.querySelector('#play').disabled,{},{timeout:30000});
  const state=await page.request.get(baseURL+'/api/status');assert.equal((await state.json()).active,null);
  assert.equal((await page.evaluate(()=>window.castleDebug())).audio.active,0,'Sounds still active after stop');
 }
 try{
  await page.goto(baseURL+'/');
  if(!process.env.MONITOR_ONLY)await run('reference',10);
  if(!process.env.REFERENCE_ONLY){
   if(!process.env.MONITOR_ONLY)await run('model',20);
   await run('model',20,true);
  }
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(output,'mobile.png')});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify({passed:true,safari,displayOnly,results,errors},null,2));
  console.log('BROWSER_PASS');
 }finally{
  await page.evaluate(()=>document.querySelector('#stop').click()).catch(()=>{});
  await browser.close();
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
