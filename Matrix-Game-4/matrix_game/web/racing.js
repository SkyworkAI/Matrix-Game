'use strict';
const RacingHud={
 labels(s){return {speed:`${Math.round(Math.abs(s.speed_kmh||0))} km/h`,mode:s.pursuit?'Pursuit':s.race?.route_name||'Free drive',
   progress:s.pursuit?`${Number(s.pursuit.capture_held_s||0).toFixed(1)} / 5 s`:s.race?`${s.race.next}/${s.race.total}`:''};},
 draw(ctx,s,w,h){
  const scale=Math.min(w/864,h/480),labels=this.labels(s);ctx.save();ctx.scale(scale,scale);w/=scale;h/=scale;
  ctx.fillStyle='#101818dd';ctx.fillRect(14,14,235,s.race?82:48);ctx.fillStyle='#fff';ctx.font='18px system-ui';ctx.fillText(labels.speed,25,42);
  if(s.race){ctx.font='13px system-ui';ctx.fillText(s.pursuit?'PURSUIT':s.race.route_name,25,63);
   let line=s.race.countdown>0?`START ${Math.ceil(s.race.countdown)}`:s.race.finished?'FINISHED':s.pursuit?
     `${Number(s.pursuit.distance_m||0).toFixed(1)} m  ${labels.progress}`:
     `${Number(s.race.elapsed||0).toFixed(2)} s  +${Number(s.race.penalty||0).toFixed(1)} s  ${labels.progress}`;
   ctx.fillStyle='#5ed79f';ctx.fillText(line,25,84);}
  const map=s.map||{},size=180,x=w-size-14,y=h-size-14,inner=size/2-9,cx=x+size/2,cy=y+size/2;
  ctx.fillStyle='#101818ee';ctx.fillRect(x,y,size,size);
  const line=(a,color,width)=>{ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(cx+a[0]*inner,cy+a[1]*inner);ctx.lineTo(cx+a[2]*inner,cy+a[3]*inner);ctx.stroke();};
  for(const a of map.roads||[])line(a,a[4]?'#93989c':'#535f65',1);
  for(const a of map.preview||[])line(a,'#86b9e9',2);
  for(const a of map.active||[])line(a,'#3c9cff',3);
  const point=(p,color,r)=>{if(!p)return;ctx.fillStyle=color;ctx.beginPath();ctx.arc(cx+Math.max(-1,Math.min(1,p[0]))*inner,cy+Math.max(-1,Math.min(1,p[1]))*inner,r,0,Math.PI*2);ctx.fill();};
  point(map.gate,'#efc458',4);point(map.suspect,'#ff6670',5);point(map.player||[0,0],'#5ed79f',4);
  ctx.fillStyle='#fff';ctx.font='11px system-ui';ctx.fillText('N',x+7,y+15);
  ctx.strokeStyle='#5ed79f';ctx.lineWidth=3;for(const a of s.gate?.segments||[]){ctx.beginPath();ctx.moveTo(a[0]*w,a[1]*h);ctx.lineTo(a[2]*w,a[3]*h);ctx.stroke();}
  ctx.restore();
 }
};
class RacingAudio{
 constructor(){this.enabled=true;this.error=null;this.ctx=null;this.buffers={};this.loops={};this.lastFrame=-1;this.impact=0;this.checkpoint=0;}
 async unlock(){
  if(!this.ctx)this.ctx=new (window.AudioContext||window.webkitAudioContext)();
  await this.ctx.resume();
  if(!this.loading)this.loading=Promise.all(['engine-loop','tire-loop','tire-front-loop','impact-medium','impact-heavy'].map(async name=>{
   const r=await fetch(`/racing-audio/${name}.wav`);if(!r.ok)throw Error('Missing racing audio');this.buffers[name]=await this.ctx.decodeAudioData(await r.arrayBuffer());
  }));await this.loading;
 }
 setEnabled(value){this.enabled=value;if(!value)this.silence();}
 loop(name,gain,rate,at){
  if(!this.buffers[name])return;let node=this.loops[name];
  if(!node){const source=this.ctx.createBufferSource(),volume=this.ctx.createGain();source.buffer=this.buffers[name];source.loop=true;volume.gain.value=0;source.connect(volume).connect(this.ctx.destination);source.start();node=this.loops[name]={source,volume};}
  node.volume.gain.setTargetAtTime(gain,at,.025);node.source.playbackRate.setTargetAtTime(rate,at,.025);
 }
 present(records,state,displayTime,visible){
  if(state.frame<=this.lastFrame)return;const previous=this.lastFrame;this.lastFrame=state.frame;
  if(!this.ctx||this.ctx.state!=='running'||!this.enabled||!visible||state.audio?.muted){this.silence();return;}
  const a=state.audio||{},at=this.ctx.currentTime+Math.max(0,(displayTime-performance.now())/1000);
  this.loop('engine-loop',.12+.17*Math.min(1,Math.abs(a.throttle||0)),Math.max(.65,Math.min(2.8,(a.rpm||900)/1800)),at);
  this.loop('tire-loop',a.rear_contacts?Math.min(.35,Math.max(0,(a.rear_slip||0)-1)*.06):0,1,at);
  this.loop('tire-front-loop',a.front_contacts?Math.min(.21,Math.max(0,(a.front_slip||0)-3)*.036):0,1,at);
  const impacts=[];
  for(const record of records){const g=record.game;if(g.frame>previous&&g.frame<=state.frame)impacts.push(...(g.audio?.impacts||[]));}
  for(const hit of impacts.sort((a,b)=>a.id-b.id)){if(hit.id<=this.impact)continue;this.impact=hit.id;const source=this.ctx.createBufferSource(),gain=this.ctx.createGain();source.buffer=this.buffers[hit.speed>5?'impact-heavy':'impact-medium'];gain.gain.value=Math.min(.5,hit.speed*.04);source.connect(gain).connect(this.ctx.destination);source.start(at);}
  if(a.checkpoint>this.checkpoint){this.checkpoint=a.checkpoint;const o=this.ctx.createOscillator(),g=this.ctx.createGain();o.frequency.value=880;g.gain.setValueAtTime(.08,at);g.gain.exponentialRampToValueAtTime(.001,at+.16);o.connect(g).connect(this.ctx.destination);o.start(at);o.stop(at+.17);}
 }
 silence(){if(this.ctx)for(const n of Object.values(this.loops))n.volume.gain.setTargetAtTime(0,this.ctx.currentTime,.025);}
 reset(){this.silence();this.lastFrame=-1;this.impact=0;this.checkpoint=0;}
 status(){return {enabled:this.enabled,ready:!!this.buffers['engine-loop'],error:this.error};}
}
