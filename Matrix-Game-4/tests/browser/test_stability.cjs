const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium, webkit} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
 const output=path.resolve(process.argv[2]);fs.mkdirSync(output,{recursive:true});
 const baseURL=process.env.TEST_URL||'http://127.0.0.1:8025';
 const seconds=Number(process.env.STABILITY_SECONDS||180);
 assert(Number.isFinite(seconds)&&seconds>=10&&seconds<=600);
 const safari=process.env.TEST_WEBKIT==='1';
 const browser=await(safari?webkit:chromium).launch({headless:true,...(!safari&&process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const context=await browser.newContext({viewport:{width:1280,height:800}});
 let page=await context.newPage();const errors=[],samples=[];
 const observe=p=>p.on('pageerror',e=>errors.push(String(e)));
 observe(page);
 async function start(){
  await page.goto(baseURL);
  await page.selectOption('#mode','model');
  await page.click('#play');
  await page.waitForFunction(()=>window.castleDebug().presented>12&&window.castleDebug().matched>0,{},{timeout:90000});
  await page.click('#surface');
  await page.waitForFunction(()=>window.castleDebug().locked,{},{timeout:5000});
 }
 try{
  await start();
  let previous=null,stalled=0;
  for(let elapsed=0;elapsed<seconds;elapsed+=5){
   await page.keyboard.down(elapsed%20<10?'w':'d');
   await page.waitForTimeout(5000);
   await page.keyboard.up('w');await page.keyboard.up('d');
   const value=await page.evaluate(()=>({at:performance.now(),...window.castleDebug(),
     metadataSize:metadata.size,inputClockSize:inputClock.size,wsBuffered:ws?.bufferedAmount||0,
     videoAgeMs:performance.now()-lastVideoAt}));
   assert(value.session,'Session ended before the stability interval completed');
   assert.equal(value.error,null);
   assert(value.metadataSize<=160&&value.inputClockSize<=512,'Client state grew beyond bounds');
   assert(value.server.media_queue_depth<=value.server.media_queue_capacity,'Unbounded media queue');
   if(previous){
    stalled=value.gameFrame>previous.gameFrame&&value.presented>previous.presented?0:stalled+1;
    assert(stalled<2,'No matched video progress for ten seconds');
    assert(value.server.frames>previous.server.frames,'Model stopped progressing');
   }
   samples.push(value);previous=value;
   console.log('STABILITY',JSON.stringify({elapsed:elapsed+5,frame:value.gameFrame,
     presented:value.presented,modelFPS:value.server.model_fps,videoAgeMs:value.videoAgeMs,
     mediaQueue:value.server.media_queue_depth,metadata:value.metadataSize}));
  }
  await page.screenshot({path:path.join(output,'final.png')});
  const session=samples.at(-1).session;
  // Simulate losing the client without its best-effort pagehide stop request.
  await context.route('**/api/*/stop',route=>route.abort());
  await page.evaluate(()=>{navigator.sendBeacon=()=>false;});
  await page.close();
  let cleaned=false;
  for(let i=0;i<30;i++){
   await new Promise(resolve=>setTimeout(resolve,1000));
   const state=await(await context.request.get(baseURL+'/api/status')).json();
   if(state.active===null){cleaned=true;break;}
   assert.equal(state.active,session);
  }
  assert(cleaned,'Disconnected session was not reclaimed');
  await context.unroute('**/api/*/stop');
  page=await context.newPage();observe(page);await start();
  await page.evaluate(()=>document.exitPointerLock());
  const stopped=page.waitForResponse(r=>r.url().endsWith('/stop')&&r.request().method()==='POST');
  await page.click('#stop');assert([200,410].includes((await stopped).status()));
  assert.equal((await(await context.request.get(baseURL+'/api/status')).json()).active,null);
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify({passed:true,safari,seconds,
    disconnectCleanup:true,restart:true,errors,samples},null,2));
  console.log('STABILITY_PASS');
 }finally{
  await context.unroute('**/api/*/stop').catch(()=>{});
  await page.evaluate(()=>document.querySelector('#stop').click()).catch(()=>{});
  await browser.close();
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
