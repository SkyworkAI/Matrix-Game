const assert = require('node:assert/strict');
const {draw,labels} = require('../../matrix_game/web/hud.js');
const state = {health:75,shield:30,weapon:1,ammo:8,action:'Reload',trap_cooldown:1.2,
 cover_status:'READY',cover_hint:'Too close to wall',cover_durability:0,
 player:1,automatic:false,raised:true,owned:[true,true,true]};
assert.deepEqual(labels(state),{health:'HP 75',shield:'SHIELD 30',weapon:'RIFLE',ammo:'8',
 action:'Reload',trap:'TRAP 2 S',cover:'COVER READY',player:'P1  MANUAL'});
assert.equal(labels({...state,trap_cooldown:0}).trap,'TRAP READY');
for(const width of [360,390,600,640,864,1280]){
 const height=width/1.8,rectangles=[];
 const ctx={fillRect(x,y,w,h){rectangles.push([x,y,w,h]);}};
 draw(ctx,state,width,height,24);
 assert(rectangles.length>100);
 for(const [x,y,w,h] of rectangles){
  assert([x,y,w,h].every(Number.isFinite));
  assert(x>=0&&y>=0&&w>=0&&h>=0&&x+w<=width+.001&&y+h<=height+.001,`HUD overflow at ${width}: ${[x,y,w,h]}`);
 }
 assert(rectangles.some(([x,y,w,h])=>x===22&&h===5&&w===(width<600?116:190)*.75),'HP bar');
}
console.log('HUD_TEST_PASSED');
