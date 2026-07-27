// Trading Strategy Hub 共用 JS（20260711 拆頁後，單一來源；各頁面皆引用本檔）
function showMain(sectionId, btn) {
  const commodity = btn.closest('.commodity-section');
  commodity.querySelectorAll('.main-section').forEach(s => s.classList.remove('active'));
  btn.closest('.commodity-subnav').querySelectorAll('.nav-main-tab').forEach(b => b.classList.remove('active'));
  document.getElementById(sectionId).classList.add('active');
  btn.classList.add('active');
}

function showTab(prefix, id, btn) {
  const panelId = prefix + '-' + id;
  btn.closest('.commodity-section').querySelectorAll('[id^="' + prefix + '-"]').forEach(p => {
    if (p.classList.contains('tab-panel')) p.classList.remove('active');
  });
  btn.closest('.subnav').querySelectorAll('.sub-tab').forEach(b => b.classList.remove('active'));
  document.getElementById(panelId).classList.add('active');
  btn.classList.add('active');
}

// 拆頁後不再需要 showCommodity（頂層改真連結）；macro_indicator fragment 內的連結仍呼叫此函式
function goToMacroBacktest() {
  const macroMainBtn = document.querySelector('[onclick*="xauusd-main-macro"]');
  if (macroMainBtn) showMain('xauusd-main-macro', macroMainBtn);
  const macroTabBtn = document.querySelector('[onclick*="macrobacktest"]');
  if (macroTabBtn) showTab('xauusd-macro', 'macrobacktest', macroTabBtn);
  window.scrollTo({top: 0, behavior: 'smooth'});
}

// 深連結：xauusd.html#xauusd-main-fvg 直接展開該主分頁
document.addEventListener('DOMContentLoaded', () => {
  const h = location.hash.replace('#', '');
  if (!h) return;
  const btn = document.querySelector('[onclick*="' + h + '"]');
  if (btn && btn.classList.contains('nav-main-tab')) showMain(h, btn);
});
