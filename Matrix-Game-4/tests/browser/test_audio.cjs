const assert=require('node:assert/strict');
const CastleAudio=require('../../matrix_game/web/audio.js');
const audio=new CastleAudio();audio.context={state:'running'};audio.buffers=Array(7).fill({});
const calls=[];audio.play=(cue)=>calls.push(cue.sound);
const cue=(sound=0)=>({sound,gain:1,pan:0,offset:-.01});
const record=(frame,sounds=[cue()])=>({game:{frame,simulation_time:frame/40,sounds}});
function present(records,current,delay=0,visible=true){audio.present(records,current.game,performance.now()-delay,visible);}
const r=record(10);assert.equal(calls.length,0,'Receiving metadata must not play audio');
present([r],r);present([r],r);assert.deepEqual(calls,[0],'Same displayed frame must not repeat audio');
present([record(11,[cue(1)]),record(12,[cue(2)])],record(12));assert.deepEqual(calls,[0,1,2],'Small display skip retains recent sounds');
present([record(13),record(21,[cue(3)])],record(21));assert.deepEqual(calls,[0,1,2,3],'Old skipped sounds must be dropped');
present([record(22)],record(22),200);assert.equal(calls.length,4,'Late metadata must not replay old sounds');
audio.enabled=false;present([record(23)],record(23));audio.enabled=true;present([record(23)],record(23));assert.equal(calls.length,4);
present([record(24)],record(24),0,false);assert.equal(calls.length,4,'Hidden page must not play');
audio.context.state='suspended';present([record(25)],record(25));assert.equal(calls.length,4);
audio.context.state='running';present([record(26,[cue(99),{...cue(),gain:NaN},{...cue(),pan:2},{...cue(),offset:NaN}])],record(26));assert.equal(calls.length,4);
audio.reset();present([record(0,[cue(6)])],record(0));assert.equal(calls.at(-1),6,'Restart must reset frame watermark');
console.log('AUDIO_SCHEDULING_PASS');
