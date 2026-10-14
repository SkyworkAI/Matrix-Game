(function(root){
'use strict';
class CastleAudio {
 constructor(){this.context=null;this.buffers=[];this.voices=new Set();this.enabled=true;this.lastFrame=-1;this.loading=null;this.error=null;this.played=0;this.dropped=0;this.bySound=Array(7).fill(0);}
 async unlock(){
  if(!this.enabled)return;
  if(!this.context){
   const Context=root.AudioContext||root.webkitAudioContext;
   if(!Context)throw Error('Web Audio is unavailable');
   this.context=new Context({latencyHint:'interactive'});
   this.mix=this.context.createGain();this.mix.gain.value=1/8;
   const limiter=this.context.createWaveShaper(),curve=new Float32Array(8193);
   for(let i=0;i<curve.length;i++)curve[i]=Math.tanh((2*i/(curve.length-1)-1)*8);
   limiter.curve=curve;this.mix.connect(limiter);limiter.connect(this.context.destination);
  }
  // Called directly from a user gesture, before any fetch or session startup.
  const resume=this.context.resume();
  if(!this.loading)this.loading=Promise.all(Array.from({length:7},async(_,i)=>{
   const response=await fetch(`/audio/${i}.wav`);if(!response.ok)throw Error(`Sound ${i}: HTTP ${response.status}`);
   return this.context.decodeAudioData(await response.arrayBuffer());
  })).then(buffers=>{this.buffers=buffers;}).catch(error=>{this.loading=null;throw error;});
  await Promise.all([resume,this.loading]);this.error=null;
 }
 setEnabled(value){this.enabled=value;if(!value)this.silence();}
 silence(){for(const voice of [...this.voices])voice.stop();}
 reset(){this.silence();this.lastFrame=-1;this.played=this.dropped=0;this.bySound.fill(0);}
 status(){return {enabled:this.enabled,state:this.context?.state||'uninitialized',ready:this.buffers.length===7,played:this.played,dropped:this.dropped,bySound:[...this.bySound],active:this.voices.size,error:this.error};}
 present(records,current,displayTime,visible=true){
  if(current.frame<=this.lastFrame)return;
  const previous=this.lastFrame;this.lastFrame=current.frame;
  const now=performance.now(),late=Math.max(0,(now-displayTime)/1000);
  const frames=[...records].filter(r=>r.game.frame>previous&&r.game.frame<=current.frame).sort((a,b)=>a.game.frame-b.game.frame);
  for(const {game} of frames)for(const cue of game.sounds||[]){
   const age=current.simulation_time-game.simulation_time-cue.offset+late;
   if(!Number.isInteger(cue.sound)||cue.sound<0||cue.sound>=7||!Number.isFinite(age)||age<0||age>.12||
      !Number.isFinite(cue.gain)||cue.gain<0||cue.gain>1||!Number.isFinite(cue.pan)||Math.abs(cue.pan)>1||
      !visible||!this.enabled||this.context?.state!=='running'||this.buffers.length!==7){this.dropped++;continue;}
   this.play(cue,displayTime,now);
  }
 }
 play(cue,displayTime,now){
  if(this.voices.size>=32)this.voices.values().next().value.stop();
  const ctx=this.context,source=ctx.createBufferSource(),left=ctx.createGain(),right=ctx.createGain(),stereo=ctx.createChannelMerger(2);
  source.buffer=this.buffers[cue.sound];
  const gain=.6*4*cue.gain; // Same .6 volume, equal-power pan and tanh mix as native PCM.
  left.gain.value=gain*Math.sqrt((1-cue.pan)*.5);right.gain.value=gain*Math.sqrt((1+cue.pan)*.5);
  source.connect(left);source.connect(right);left.connect(stereo,0,0);right.connect(stereo,0,1);stereo.connect(this.mix);
  let stopped=false;
  const voice={stop:()=>{if(stopped)return;stopped=true;source.stop();source.disconnect();left.disconnect();right.disconnect();stereo.disconnect();this.voices.delete(voice);}};
  source.onended=voice.stop;this.voices.add(voice);
  const latency=(ctx.outputLatency||0)+(ctx.baseLatency||0);
  source.start(ctx.currentTime+Math.max(0,(displayTime-now)/1000-latency));
  this.played++;this.bySound[cue.sound]++;
 }
}
root.CastleAudio=CastleAudio;
if(typeof module!=='undefined')module.exports=CastleAudio;
})(typeof window!=='undefined'?window:globalThis);
