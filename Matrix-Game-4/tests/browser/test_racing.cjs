const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const output = path.resolve(process.argv[2]);
  fs.mkdirSync(output, {recursive:true});
  const baseURL = process.env.TEST_URL || 'http://127.0.0.1:8025';
  const seconds = Number(process.env.TEST_SECONDS || 60);
  const mode = process.env.TEST_MODE || 'reference';
  const raceModes = (process.env.RACE_MODES || 'time_trial,pursuit').split(',');
  assert(Number.isFinite(seconds) && seconds >= 5 && seconds <= 300);
  const browser = await chromium.launch({headless:true,
    ...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {})});
  const page = await browser.newPage({viewport:{width:1280,height:800}});
  const errors=[], runs=[];
  page.on('pageerror', error => errors.push(String(error)));
  try {
    await page.goto(baseURL);
    for (const raceMode of raceModes) {
      await page.goto(baseURL);
      await page.selectOption('#mode', mode);
      await page.selectOption('#race-mode', raceMode);
      await page.click('#play');
      await page.waitForFunction(() => window.castleDebug().presented > 12 && window.castleDebug().matched > 0,
        {}, {timeout:90000});
      await page.click('#surface');
      await page.waitForFunction(() => window.castleDebug().locked, {}, {timeout:5000});
      const samples=[];
      let previous=null;
      for (let elapsed=0; elapsed<seconds; elapsed+=5) {
        const drive=process.env.TEST_SHUTTLE==='1' && elapsed%20>=10?'s':'w';
        await page.keyboard.down(drive);
        if (process.env.TEST_SHUTTLE!=='1' && elapsed % 20 === 10) await page.keyboard.down('d');
        if (process.env.TEST_SHUTTLE!=='1' && elapsed % 20 === 15) await page.keyboard.down('a');
        await page.waitForTimeout(5000);
        await page.keyboard.up('d'); await page.keyboard.up('a');await page.keyboard.up(drive);
        const sample = await page.evaluate(() => ({at:performance.now(), ...window.castleDebug(),
          metadataSize:metadata.size,inputClockSize:inputClock.size,videoAgeMs:performance.now()-lastVideoAt,
          wsBuffered:ws?.bufferedAmount||0}));
        samples.push(sample);
        fs.writeFileSync(path.join(output,`${raceMode}-samples.json`),JSON.stringify(samples,null,2));
        assert(sample.session, 'Session ended unexpectedly');
        assert.equal(sample.error, null);
        assert(sample.audio.ready, 'Vehicle audio assets failed to decode');
        assert(sample.metadataSize<=160 && sample.inputClockSize<=512);
        assert(sample.server.media_queue_depth<=sample.server.media_queue_capacity);
        if (previous) {
          assert(sample.server.frames>previous.server.frames, 'Generator stalled');
          assert(sample.presented>previous.presented && sample.gameFrame>previous.gameFrame, 'Display stalled');
        }
        previous=sample;
        console.log('RACING', JSON.stringify({raceMode,elapsed:elapsed+5,frame:sample.gameFrame,
          onlineFPS:sample.server.online_fps,modelFPS:sample.server.model_fps,
          promptUpdates:sample.server.prompt_updates,promptEncodes:sample.server.prompt_encodes,
          presented:sample.presented,videoAgeMs:sample.videoAgeMs}));
        if (elapsed % 20 === 15 && raceMode !== 'drive') await page.keyboard.press('r');
      }
      await page.keyboard.up('w');
      const pixels = await page.evaluate(() => {
        const c=document.createElement('canvas');c.width=96;c.height=54;
        const x=c.getContext('2d');x.drawImage(document.querySelector('video'),0,0,96,54);
        const data=x.getImageData(0,0,96,54).data;
        let sum=0,squared=0,count=0;
        for(let i=0;i<data.length;i+=4){const v=(data[i]+data[i+1]+data[i+2])/3;sum+=v;squared+=v*v;count++;}
        return {mean:sum/count,variance:squared/count-(sum/count)**2};
      });
      assert(pixels.mean>3 && pixels.variance>10, 'Blank video');
      await page.screenshot({path:path.join(output,`${raceMode}-desktop.png`)});
      await page.evaluate(() => document.exitPointerLock());
      await page.setViewportSize({width:390,height:844});
      await page.waitForTimeout(500);
      await page.screenshot({path:path.join(output,`${raceMode}-mobile.png`)});
      assert(await page.evaluate(() => document.documentElement.scrollWidth<=innerWidth), 'Horizontal overflow');
      const stopped=page.waitForResponse(r=>r.url().endsWith('/stop')&&r.request().method()==='POST');
      await page.click('#stop');
      assert([200,410].includes((await stopped).status()));
      await page.waitForFunction(() => !window.castleDebug().session);
      assert.equal((await(await page.request.get(baseURL+'/api/status')).json()).active,null);
      runs.push({raceMode,pixels,samples});
      if (mode==='model' && process.env.EXPECT_DYNAMIC==='1')
        assert(samples.at(-1).server.prompt_updates>1, 'Dynamic prompts did not change');
      fs.writeFileSync(path.join(output,'report.json'), JSON.stringify({mode,seconds,errors,runs},null,2));
      await page.setViewportSize({width:1280,height:800});
      await page.waitForTimeout(2000);
    }
    assert.deepEqual(errors,[]);
    console.log('RACING_PASS');
  } finally {
    await page.evaluate(() => document.querySelector('#stop').click()).catch(()=>{});
    await browser.close();
  }
})().catch(error => {console.error(error);process.exitCode=1;});
