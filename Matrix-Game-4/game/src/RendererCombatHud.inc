void Renderer::drawCombatOverlay(const HudState& hud) {
    if(hud.showHelp){auto hints=CombatUi::helpLines(width);
    float footer=CombatUi::footerHeight(width),y=float(height)-footer;
    rectangle(0,y,float(width),footer,{.025f,.025f,.025f,.82f});
    for(size_t i=0;i<hints.size();++i)
        text(16,y+4+i*CombatUi::HelpLineHeight,hints[i],CombatUi::HelpScale,rgb(209,216,211));}
    if(!hud.paused)return;
    rectangle(0,0,float(width),float(height),{0,0,0,.66f});
    auto panel=CombatUi::menuPanel(width,height);
    rectangle(panel.x,panel.y,panel.w,panel.h,{.035f,.04f,.038f,1});
    text(panel.x+16,panel.y+13,hud.dead?"GAME OVER":"PAUSED",3,rgb(231,236,232));
    for(int i=0;i<int(hud.dead?CombatUi::DeathLabels.size():CombatUi::MenuLabels.size());++i) {
        auto item=CombatUi::menuItem(width,height,i);
        if(i==hud.pauseSelection)rectangle(item.x,item.y,item.w,item.h,{.15f,.25f,.21f,1});
        text(item.x+8,item.y+7,hud.dead?CombatUi::DeathLabels[i]:CombatUi::MenuLabels[i],1.75f,rgb(231,236,232));
        if(!hud.dead&&i==hud.weapon+1)text(item.x+item.w-66,item.y+9,"ACTIVE",1.25f,rgb(94,215,159));
    }
    text(panel.x+16,panel.y+195,hud.dead?"CLICK / UP DOWN ENTER":"CLICK / UP DOWN ENTER / ESC",1.25f,rgb(185,195,187));
}
