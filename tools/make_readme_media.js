'use strict';
// Renders the README's pictures from the project page itself, docs/index.html: the hero, the results table, the cost
// panels, the three steps and the lead chart as PNG, and the film as GIF. Each comes at the README's width on github.com
// (838 CSS px in a 1280 px window, 324 in a 390 px phone) at 2x, so the README shows the page's own panels at the
// page's own type size. Writes them with MANIFEST.json to docs/assets/readme_page/; tests/ci/test_readme_headline.py
// checks that the README names exactly these files and that they were made from the current docs/data/headline.json.
// Needs Node 18+, playwright-core with a Chromium build, and ffmpeg; the page loads its web fonts from Google Fonts.
//   node tools/make_readme_media.js [--only hero,film] [--out DIR]
//   env: PLAYWRIGHT_CORE (module, default playwright-core), CHROMIUM (browser executable), FFMPEG (default ffmpeg)
const fs = require('fs'), os = require('os'), path = require('path'), http = require('http'), crypto = require('crypto');
const { execFileSync } = require('child_process');
const { chromium } = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');

const ROOT = path.resolve(__dirname, '..'), DOCS = path.join(ROOT, 'docs');
const argv = process.argv.slice(2);
function opt(name) { const i = argv.indexOf('--' + name); return i >= 0 ? argv[i + 1] : null; }
const OUT = path.resolve(opt('out') || path.join(DOCS, 'assets', 'readme_page'));
const ONLY = opt('only') ? opt('only').split(',') : null;
const FFMPEG = process.env.FFMPEG || 'ffmpeg';
const WIDTH = { desk: 838, phone: 324 }, SCALE = 2;

// Every picture is one element of the page, captured on a transparent ground so its rounded corners sit on either
// GitHub theme. vw: the window width that gives the element the README's width (the page keeps 16 px gutters); the
// results table and the steps need a wider window to keep their desktop layout, and are shown scaled to 838. An axis
// label that would touch the one before it on its row is left out (at 324 the build chart's "1 min" meets "10 s").
const KEEP_OUT = 'html,body{background:transparent!important;scroll-behavior:auto!important}nav,.film-ctl,#filmRows{display:none!important}';
const SHOTS = [
  { name: 'hero', sel: 'header.paper', vw: { desk: 838, phone: 324 },
    hide: '.define,.btns,.install-hero,.release-links,.byline',
    css: 'header.paper{border-radius:16px;background:radial-gradient(60% 80% at 25% 20%,rgba(68,149,234,.10),transparent 70%),var(--bg)!important}' },
  // the phone keeps the summary columns where MUSTER's chips sit (mean rank, mean score, top 3); the README's text table
  // under the picture has every cell
  { name: 'results', sel: '#t8', up: '.panel', vw: { desk: 962, phone: 356 },
    css: '#t8 th,#t8 td{padding-left:7px;padding-right:7px}',
    phone: '#t8{min-width:0!important}#t8 thead tr:first-child th.grp,#t8 thead tr:first-child th:nth-child(7),' +
           '#t8 thead tr:first-child th:nth-child(9),#t8 thead tr:nth-child(2),#t8 tbody td:nth-of-type(-n+6),' +
           '#t8 tbody td:nth-of-type(9),#t8 tbody td:nth-of-type(11),#t8note{display:none}' },
  { name: 'cost', sel: '.cost-row', vw: { desk: 870, phone: 356 } },
  { name: 'steps', sel: '#flow', up: '.panel', vw: { desk: 1000, phone: 356 } },
  { name: 'lead', sel: '#leadBars', up: '.panel', vw: { desk: 870, phone: 356 }, open: '#lead', hide: '#leadBars+details',
    css: '#lead .panel:has(#leadBars){padding:22px 24px 12px}' },
];

function sha256(file) { return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'); }
function dims(file) {
  const b = fs.readFileSync(file);
  return file.endsWith('.gif') ? [b.readUInt16LE(6), b.readUInt16LE(8)] : [b.readUInt32BE(16), b.readUInt32BE(20)];
}
function serve() {
  const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
                  '.svg': 'image/svg+xml', '.png': 'image/png', '.gif': 'image/gif', '.ico': 'image/x-icon' };
  return new Promise(done => {
    const srv = http.createServer((req, res) => {
      let p = decodeURIComponent(new URL(req.url, 'http://local').pathname);
      if (p === '/readme-film.html') {
        // the page with the film slowed SLOW times and a hold that never ends during the recording
        const T = filmTiming();
        res.writeHead(200, { 'content-type': types['.html'] });
        res.end(fs.readFileSync(path.join(DOCS, 'index.html'), 'utf8').replace(T.decl, 'PACE = ' + T.pace * SLOW + ', HOLD = 3600000;'));
        return;
      }
      if (p.endsWith('/')) p += 'index.html';
      const f = path.join(DOCS, path.normalize(p));
      if (!f.startsWith(DOCS + path.sep) || !fs.existsSync(f) || !fs.statSync(f).isFile()) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' });
      fs.createReadStream(f).pipe(res);
    }).listen(0, '127.0.0.1', () => done(srv));
  });
}
async function open(browser, url, vw, motion) {
  const ctx = await browser.newContext({ viewport: { width: vw, height: 1000 }, deviceScaleFactor: SCALE, colorScheme: 'dark', reducedMotion: motion });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto(url, { waitUntil: 'networkidle', timeout: 90000 });
  await page.evaluate(() => document.fonts.ready);
  const fonts = await page.evaluate(() => ['Inter', 'Inter Tight', 'JetBrains Mono'].filter(f => !document.fonts.check('16px "' + f + '"')));
  if (fonts.length) throw new Error('web fonts not loaded: ' + fonts.join(', '));
  return { ctx, page, errors };
}

async function shoot(browser, url, s, kind, vw) {
  vw = vw || s.vw[kind];
  const { ctx, page, errors } = await open(browser, url, vw, 'reduce');
  await page.addStyleTag({ content: KEEP_OUT + (s.css || '') + (kind === 'phone' && s.phone ? s.phone : '') + (s.hide ? s.hide + '{display:none!important}' : '') });
  if (s.open) await page.evaluate(sel => { document.querySelector(sel).open = true; }, s.open);
  await page.waitForTimeout(1200);
  const thinned = await page.evaluate(([sel, up]) => {
    let e = document.querySelector(sel);
    if (up) e = e.closest(up);
    const out = [], rows = new Map(), svgs = [...e.querySelectorAll('svg')];
    for (const t of e.querySelectorAll('svg text.c-axis')) {
      const key = svgs.indexOf(t.ownerSVGElement) + '/' + t.getAttribute('y');
      if (!rows.has(key)) rows.set(key, []);
      rows.get(key).push(t);
    }
    for (const row of rows.values()) {
      let right = -Infinity;
      for (const t of row.sort((a, b) => a.getBBox().x - b.getBBox().x)) {
        const b = t.getBBox();
        if (b.x < right + 3) { t.style.display = 'none'; out.push(t.textContent); } else right = b.x + b.width;
      }
    }
    return out;
  }, [s.sel, s.up || null]);
  const box = await page.evaluate(([sel, up]) => {
    let e = document.querySelector(sel);
    if (up) e = e.closest(up);
    // a table or chart cut by its own scroll box would show as cut in the picture: widen the window by the overflow
    const over = Math.max(0, ...[e, ...e.querySelectorAll('*')].filter(x => getComputedStyle(x).overflowX === 'auto').map(x => x.scrollWidth - x.clientWidth));
    const r = e.getBoundingClientRect();
    return { x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height, over };
  }, [s.sel, s.up || null]);
  if (box.over > 1) {
    await ctx.close();
    if (vw - s.vw[kind] > 200) throw new Error(s.name + ' ' + kind + ': still scrolls sideways at a ' + vw + ' px window');
    return shoot(browser, url, s, kind, vw + Math.ceil(box.over) + 2);
  }
  if (errors.length) throw new Error(s.name + ' ' + kind + ': page error ' + errors[0]);
  const clip = { x: Math.floor(box.x), y: Math.floor(box.y), width: Math.ceil(box.x + box.width) - Math.floor(box.x), height: Math.ceil(box.y + box.height) - Math.floor(box.y) };
  const file = path.join(OUT, s.name + (kind === 'phone' ? '_phone' : '') + '.png');
  await page.screenshot({ path: file, clip, fullPage: true, omitBackground: true });
  await ctx.close();
  console.log(s.name, kind, 'window', vw, 'element', clip.width + 'x' + clip.height + (thinned.length ? ', left out ' + thinned.join(', ') : ''));
  return file;
}

// The film runs on timers and CSS transitions. It is recorded in slow motion: the page is served with its pace slowed
// SLOW times (and a hold long enough that it never moves on by itself) and its CSS clock slowed the same, so each 2x
// screenshot costs little film time; every frame is stamped with its film time. The page shows one store per cycle at
// these widths, so the GIF plays store A, holds its last frame as the page does, then store B, as the page plays them.
// Opaque frames (square corners) keep ffmpeg's frame-to-frame transparency, which is what keeps the GIF small.
const SLOW = 8, TAIL = 1200;
const FILM = {
  // the tablet layout at its own width (552), shown at that width on a desktop
  desk: { vw: 640, layout: 't', clip: '.film-frame', css: '' },
  // the phone layout across the full phone width
  phone: { vw: 324, layout: 'p', clip: '#top', css: '.film-frame{padding-left:0!important;padding-right:0!important}' },
};
function filmTiming() {
  const src = fs.readFileSync(path.join(DOCS, 'index.html'), 'utf8');
  const m = src.match(/PACE = ([\d.]+), HOLD = (\d+);/), e = src.match(/(\d+) \* PACE \+ HOLD\)/);
  if (!m || !e || src.split(m[0]).length !== 2) throw new Error('film timing not found once in docs/index.html');
  return { pace: +m[1], hold: +m[2], last: +e[1] * +m[1], decl: m[0] };
}
async function film(browser, base, kind, tmp) {
  const F = FILM[kind], T = filmTiming(), cycle = T.last + T.hold;
  const { ctx, page, errors } = await open(browser, base + 'readme-film.html', F.vw, 'no-preference');
  await page.addStyleTag({ content: KEEP_OUT + 'header.paper,footer,main>:not(#top){display:none!important}' + F.css });
  await page.waitForTimeout(800);
  const h = await page.evaluate(() => Math.ceil(document.getElementById('top').getBoundingClientRect().height));
  await page.setViewportSize({ width: F.vw, height: h + 40 });
  await page.waitForTimeout(800);
  const layout = await page.evaluate(() => document.getElementById('film').dataset.layout);
  if (layout !== F.layout) throw new Error('film ' + kind + ': layout ' + layout + ', expected ' + F.layout);
  const clip = await page.evaluate(([c]) => {
    const x = document.querySelector(c).getBoundingClientRect(), y = document.getElementById('top').getBoundingClientRect();
    return { x: Math.floor(x.left), y: Math.floor(y.top), width: Math.round(x.width), height: Math.round(y.height) };
  }, [F.clip]);
  const cdp = await ctx.newCDPSession(page);
  await cdp.send('Animation.enable');
  await cdp.send('Animation.setPlaybackRate', { playbackRate: 1 / SLOW });
  const dir = fs.mkdtempSync(path.join(tmp, 'film-')), frames = [];
  for (const row of [0, 1]) {
    // the store's own button shows it and plays it from the first step
    const t0 = await page.evaluate(r => { document.querySelectorAll('#filmRows button')[r].click(); return performance.now(); }, row);
    for (let t = 0; t <= T.last + TAIL;) {
      const a = await page.evaluate(() => performance.now());
      const name = 'f' + String(frames.length).padStart(5, '0') + '.png';
      await page.screenshot({ path: path.join(dir, name), clip });
      const b = await page.evaluate(() => performance.now());
      t = ((a + b) / 2 - t0) / SLOW;
      frames.push({ name, t: row * cycle + (frames.length && frames[frames.length - 1].t >= row * cycle ? t : 0) });
    }
  }
  if (errors.length) throw new Error('film ' + kind + ': page error ' + errors[0]);
  const dur = frames.map((f, i) => {
    const next = i + 1 < frames.length ? frames[i + 1].t : 2 * cycle;
    // the last frame of store A holds until store B starts, at one cycle
    return (f.t < cycle && next >= cycle ? cycle : next) - f.t;
  });
  // the GIF opens on store A's finished frame and loops the same way, so a GIF paused on its first frame, or any still
  // of it, shows store A's whole answer
  const k = frames.findIndex(f => f.t >= cycle) - 1, idx = [...frames.keys()], order = idx.slice(k).concat(idx.slice(0, k));
  const list = ['ffconcat version 1.0'];
  for (const i of order) list.push("file '" + frames[i].name + "'", 'duration ' + (dur[i] / 1000).toFixed(4));
  list.push("file '" + frames[order[order.length - 1]].name + "'");
  fs.writeFileSync(path.join(dir, 'list.txt'), list.join('\n') + '\n');
  const file = path.join(OUT, 'film' + (kind === 'phone' ? '_phone' : '') + '.gif');
  execFileSync(FFMPEG, ['-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', path.join(dir, 'list.txt'), '-vf',
    'fps=25/2,format=rgb24,split[a][b];[a]palettegen=max_colors=256[p];[b][p]paletteuse=dither=none:diff_mode=rectangle',
    '-loop', '0', file]);
  await ctx.close();
  console.log('film', kind, layout, 'cycle', cycle, 'ms,', frames.length, 'frames, clip', clip.width + 'x' + clip.height);
  return file;
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'readme-media-'));
  const srv = await serve(), url = 'http://127.0.0.1:' + srv.address().port + '/';
  const browser = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
  const made = [];
  try {
    for (const s of SHOTS) for (const kind of ['desk', 'phone']) if (!ONLY || ONLY.includes(s.name)) made.push(await shoot(browser, url, s, kind));
    for (const kind of ['desk', 'phone']) if (!ONLY || ONLY.includes('film')) made.push(await film(browser, url, kind, tmp));
  } finally {
    await browser.close();
    srv.close();
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  const files = {};
  for (const f of made) {
    const [w, h] = dims(f);
    files[path.basename(f)] = { sha256: sha256(f), bytes: fs.statSync(f).size, pixels: w + 'x' + h };
    console.log(path.basename(f), files[path.basename(f)].pixels, Math.round(files[path.basename(f)].bytes / 1024) + ' KB');
  }
  if (ONLY) { console.log('--only: MANIFEST.json not written'); return; }
  const rel = p => path.relative(ROOT, p).split(path.sep).join('/');
  const manifest = {
    about: 'Pictures of the project page for the README, made by tools/make_readme_media.js from the inputs below. Run it again after the page or the headline data changes.',
    inputs: Object.fromEntries([path.join(DOCS, 'index.html'), path.join(DOCS, 'data', 'headline.json')].map(p => [rel(p), sha256(p)])),
    css_width: WIDTH, scale: SCALE, files,
  };
  fs.writeFileSync(path.join(OUT, 'MANIFEST.json'), JSON.stringify(manifest, null, 2) + '\n');
})().catch(e => { console.error(e.stack || String(e)); process.exit(1); });
