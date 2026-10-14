'use strict';
// Same 5x7 font, colors and combat layout as the native RendererHudLinux.inc.
const CastleHud = (() => {
 const font = {
  '0':[14,17,19,21,25,17,14],'1':[4,12,4,4,4,4,14],'2':[14,17,1,2,4,8,31],
  '3':[30,1,1,14,1,1,30],'4':[2,6,10,18,31,2,2],'5':[31,16,16,30,1,1,30],
  '6':[6,8,16,30,17,17,14],'7':[31,1,2,4,8,8,8],'8':[14,17,17,14,17,17,14],
  '9':[14,17,17,15,1,2,12],'A':[14,17,17,31,17,17,17],'B':[30,17,17,30,17,17,30],
  'C':[14,17,16,16,16,17,14],'D':[30,17,17,17,17,17,30],'E':[31,16,16,30,16,16,31],
  'F':[31,16,16,30,16,16,16],'G':[14,17,16,23,17,17,15],'H':[17,17,17,31,17,17,17],
  'I':[14,4,4,4,4,4,14],'J':[7,2,2,2,2,18,12],'K':[17,18,20,24,20,18,17],
  'L':[16,16,16,16,16,16,31],'M':[17,27,21,21,17,17,17],'N':[17,25,21,19,17,17,17],
  'O':[14,17,17,17,17,17,14],'P':[30,17,17,30,16,16,16],'Q':[14,17,17,17,21,18,13],
  'R':[30,17,17,30,20,18,17],'S':[15,16,16,14,1,1,30],'T':[31,4,4,4,4,4,4],
  'U':[17,17,17,17,17,17,14],'V':[17,17,17,17,17,10,4],'W':[17,17,17,21,21,27,17],
  'X':[17,17,10,4,10,17,17],'Y':[17,17,10,4,4,4,4],'Z':[31,1,2,4,8,16,31],
  '.':[0,0,0,0,0,6,6],'-':[0,0,0,31,0,0,0],'/':[1,1,2,4,8,16,16],
  ':':[0,6,6,0,6,6,0],'%':[17,2,4,4,8,16,17]
 };
 const names=['PISTOL','RIFLE','SHOTGUN'];
 function labels(g) {
  return {health:`HP ${Math.trunc(g.health)}`,shield:`SHIELD ${Math.trunc(g.shield)}`,
   weapon:names[g.weapon]||'--',ammo:String(g.ammo),action:g.action||'',
   trap:`TRAP ${g.trap_cooldown>0?Math.ceil(g.trap_cooldown)+' S':'READY'}`,
   cover:`COVER ${g.cover_status||'READY'}`,player:`P${g.player||1}  ${g.automatic?'AUTO':'MANUAL'}`};
 }
 function draw(ctx,g,width,height,fps) {
  const white='#e7ece8',green='#5ed79f',panel='rgba(6,6,6,.8)',l=labels(g);
  const rect=(x,y,w,h,color)=>{ctx.fillStyle=color;ctx.fillRect(x,y,w,h);};
  function text(x,y,value,scale,color,maxWidth=Infinity) {
   const chars=String(value).toUpperCase();scale=Math.min(scale,maxWidth/Math.max(1,chars.length*6));
   for(const ch of chars){const rows=font[ch];if(rows)rows.forEach((bits,r)=>{for(let c=0;c<5;c++)if(bits&(1<<(4-c)))rect(x+c*scale,y+r*scale,scale,scale,color);});x+=6*scale;}
  }
  const compact=width<600, y=height-80, leftWidth=compact?136:210;
  const weaponWidth=compact?width-172:246, weaponX=width-weaponWidth-12;
  rect(12,12,156,56,panel);text(22,22,`FPS ${fps>0?fps.toFixed(1):'--'}`,2,white);
  text(22,44,fps>0?`${(1000/fps).toFixed(2)} MS`:'-- MS',2,white);
  text(width-(compact?168:240),22,l.player,2,white);
  const cx=width/2,cy=height/2;
  for(const [x,z,w,h] of [[cx-12,cy-.75,7,1.5],[cx+5,cy-.75,7,1.5],[cx-.75,cy-12,1.5,7],[cx-.75,cy+5,1.5,7]])rect(x,z,w,h,white);
  rect(12,y,leftWidth,62,panel);text(22,y+10,l.health,compact?1.5:2,white);
  rect(22,y+30,leftWidth-20,5,'#333');rect(22,y+30,(leftWidth-20)*Math.max(0,Math.min(1,g.health/100)),5,green);
  text(22,y+44,l.shield,compact?1.2:1.5,g.raised?'#69b1ff':white);
  rect(weaponX,y,weaponWidth,62,panel);text(weaponX+12,y+10,l.weapon,compact?1.5:2,white,weaponWidth-68);
  text(width-64,y+10,l.ammo,compact?2:3,white,52);
  text(weaponX+12,y+36,l.action,1.5,green,weaponWidth-24);
  text(weaponX+12,y+50,l.trap,1.2,white,weaponWidth-24);
  rect(weaponX,y-30,weaponWidth,26,panel);
  text(weaponX+12,y-22,l.cover,1.3,g.cover_durability>0?green:white,weaponWidth-24);
  if(g.cover_hint)text(weaponX,y-45,g.cover_hint,1.2,'#ffc878',weaponWidth);
  if(width>=850)for(let i=0;i<3;i++){
   const x=cx-60+i*42,owned=g.owned?.[i];
   rect(x,y+25,34,3,owned?(i===g.weapon?white:green):'#424242');
   text(x+12,y+8,String(i+1),1.6,owned?white:'#585858');
  }
 }
 return {draw,labels};
})();
if(typeof module!=='undefined')module.exports=CastleHud;
