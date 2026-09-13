/* 资料问答工作台 · 前端逻辑
   原生 JavaScript，无依赖。所有来自后端或用户的文本一律经 textContent / DOM 节点写入，
   不使用 innerHTML 渲染不可信内容。 */
'use strict';

(() => {
  const POLL_MS = 1800;
  const HEALTH_RETRY_MS = 3000;
  const TEST_TIMEOUT_MS = 60000;
  const DRAFT_KEY = 'rag-demo.draft';
  const SUBMISSION_KEY = 'rag-demo.pending-submission';
  const PENDING = new Set(['queued', 'running']);
  const STATUS_LABEL = {
    queued: '排队中',
    running: '生成中',
    answered: '已回答',
    no_answer: '未找到答案',
    failed: '失败',
    interrupted: '已中断',
  };
  const LANGUAGES = ['auto', 'zh', 'en'];
  const DETAILS = ['concise', 'balanced', 'detailed'];
  const SVG_NS = 'http://www.w3.org/2000/svg';

  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------------------
  // DOM 工具
  // ---------------------------------------------------------------------------

  function el(tag, props, ...children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (value === undefined || value === null || value === false) continue;
        if (key === 'class') node.className = value;
        else if (key === 'text') node.textContent = value;
        else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
        else if (key === 'dataset') Object.assign(node.dataset, value);
        else node.setAttribute(key, value === true ? '' : String(value));
      }
    }
    append(node, children);
    return node;
  }

  function append(node, children) {
    for (const child of children) {
      if (child === null || child === undefined || child === false) continue;
      if (Array.isArray(child)) append(node, child);
      else if (child instanceof Node) node.appendChild(child);
      else node.appendChild(document.createTextNode(String(child)));
    }
  }

  function icon(d, cls) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    if (cls) svg.setAttribute('class', cls);
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', d);
    svg.appendChild(path);
    return svg;
  }

  const ICON = {
    external: 'M14 4h6v6M20 4l-9 9M18 14v6H4V6h6',
    doc: 'M6 3h8l4 4v14H6z M14 3v4h4 M9 12h6M9 15h6M9 18h4',
    zoom: 'M4 10V4h6M4 4l6 6M20 14v6h-6M20 20l-6-6',
    chev: 'M9 6l6 6-6 6',
    retry: 'M20 12a8 8 0 1 1-2.3-5.7M20 4v5h-5',
  };

  function pad2(n) { return n < 10 ? `0${n}` : String(n); }

  function formatTime(value) {
    if (!value) return '';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString('zh-CN', { hour12: false });
  }

  /** 只接受绝对 http(s) 链接，用于外部来源。 */
  function safeHttpUrl(raw) {
    if (typeof raw !== 'string' || !raw.trim()) return null;
    try {
      const url = new URL(raw.trim());
      return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : null;
    } catch {
      return null;
    }
  }

  /** 图片只允许本站地址（与 CSP img-src 'self' 一致）。 */
  function sameOriginUrl(raw) {
    if (typeof raw !== 'string' || !raw.trim()) return null;
    try {
      const url = new URL(raw.trim(), window.location.href);
      if (url.origin !== window.location.origin) return null;
      return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : null;
    } catch {
      return null;
    }
  }

  function str(value) {
    return typeof value === 'string' ? value : value === null || value === undefined ? '' : String(value);
  }

  // ---------------------------------------------------------------------------
  // HTTP
  // ---------------------------------------------------------------------------

  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.status = status;
    }
  }

  async function api(path, { method = 'GET', body, timeout = 20000, signal, headers = {} } = {}) {
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
    const onAbort = () => controller.abort();
    if (signal) signal.addEventListener('abort', onAbort, { once: true });
    try {
      let response;
      try {
        response = await fetch(path, {
          method,
          headers: { Accept: 'application/json', ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...headers },
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: controller.signal,
          cache: 'no-store',
          credentials: 'same-origin',
        });
      } catch (err) {
        if (signal && signal.aborted) throw new ApiError('请求已取消', 0);
        throw new ApiError(timedOut ? '请求超时，请稍后重试' : '无法连接本地服务，请确认后端已启动', 0);
      }
      let data = null;
      try {
        data = await response.json();
      } catch {
        data = null;
      }
      if (!response.ok) {
        const message = data && typeof data.error === 'string' && data.error ? data.error : `请求失败（HTTP ${response.status}）`;
        throw new ApiError(message, response.status);
      }
      if (data === null || typeof data !== 'object') throw new ApiError('服务返回了无法解析的数据', response.status);
      return data;
    } finally {
      clearTimeout(timer);
      if (signal) signal.removeEventListener('abort', onAbort);
    }
  }

  // ---------------------------------------------------------------------------
  // 安全 Markdown 渲染（只生成 DOM 节点）
  // ---------------------------------------------------------------------------

  const MD = {
    fence: /^ {0,3}(`{3,}|~{3,})\s*([^`\s]*)[^`]*$/,
    heading: /^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$/,
    hr: /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/,
    list: /^([ \t]*)([-*+]|\d{1,9}[.)])[ \t]+(.*)$/,
    quote: /^ {0,3}>[ \t]?(.*)$/,
    tableSep: /^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$/,
  };

  function indentOf(text) {
    let width = 0;
    for (const ch of text) {
      if (ch === ' ') width += 1;
      else if (ch === '\t') width += 4;
      else break;
    }
    return width;
  }

  const isBlank = (line) => /^\s*$/.test(line);

  function isTableStart(lines, i) {
    return i + 1 < lines.length && lines[i].includes('|') && lines[i + 1].includes('-') && MD.tableSep.test(lines[i + 1]);
  }

  function startsBlock(lines, i) {
    const line = lines[i];
    return MD.fence.test(line) || MD.heading.test(line) || MD.hr.test(line) || MD.quote.test(line) || MD.list.test(line) || isTableStart(lines, i);
  }

  function renderMarkdown(source) {
    const lines = str(source).replace(/\r\n?/g, '\n').split('\n');
    return renderBlocks(lines, 0);
  }

  function renderBlocks(lines, depth) {
    const frag = document.createDocumentFragment();
    if (depth > 12) {
      frag.appendChild(el('p', { text: lines.join('\n') }));
      return frag;
    }
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (isBlank(line)) { i += 1; continue; }

      const fence = MD.fence.exec(line);
      if (fence) {
        const marker = fence[1];
        const body = [];
        i += 1;
        while (i < lines.length) {
          const close = lines[i].trim();
          if (close.length >= marker.length && close[0] === marker[0] && /^(`+|~+)$/.test(close)) { i += 1; break; }
          body.push(lines[i]);
          i += 1;
        }
        const code = el('code', { text: body.join('\n') });
        if (fence[2]) code.dataset.lang = fence[2].slice(0, 32);
        frag.appendChild(el('pre', null, code));
        continue;
      }

      const heading = MD.heading.exec(line);
      if (heading) {
        const level = Math.min(heading[1].length + 1, 5);
        const h = el(`h${level}`);
        h.appendChild(renderInline(heading[2]));
        frag.appendChild(h);
        i += 1;
        continue;
      }

      if (MD.hr.test(line)) {
        frag.appendChild(el('hr'));
        i += 1;
        continue;
      }

      if (MD.quote.test(line)) {
        const inner = [];
        while (i < lines.length && !isBlank(lines[i])) {
          const q = MD.quote.exec(lines[i]);
          if (!q && startsBlock(lines, i)) break;
          inner.push(q ? q[1] : lines[i]);
          i += 1;
        }
        const bq = el('blockquote');
        bq.appendChild(renderBlocks(inner, depth + 1));
        frag.appendChild(bq);
        continue;
      }

      if (isTableStart(lines, i)) {
        const [table, next] = parseTable(lines, i);
        frag.appendChild(table);
        i = next;
        continue;
      }

      if (MD.list.test(line)) {
        const [list, next] = parseList(lines, i, depth);
        frag.appendChild(list);
        i = next;
        continue;
      }

      const para = [line];
      i += 1;
      while (i < lines.length && !isBlank(lines[i]) && !startsBlock(lines, i)) {
        para.push(lines[i]);
        i += 1;
      }
      const p = el('p');
      p.appendChild(renderInline(para.map((l) => l.trim()).join('\n')));
      frag.appendChild(p);
    }
    return frag;
  }

  function parseList(lines, start, depth) {
    const first = MD.list.exec(lines[start]);
    const baseIndent = indentOf(first[1]);
    const ordered = /\d/.test(first[2]);
    const list = el(ordered ? 'ol' : 'ul');
    if (ordered) {
      const n = parseInt(first[2], 10);
      if (n !== 1 && Number.isFinite(n)) list.setAttribute('start', String(n));
    }
    let i = start;
    while (i < lines.length) {
      const m = MD.list.exec(lines[i]);
      if (!m || indentOf(m[1]) !== baseIndent || /\d/.test(m[2]) !== ordered) break;
      const content = [m[3]];
      const children = [];
      i += 1;
      while (i < lines.length) {
        const line = lines[i];
        if (isBlank(line)) {
          let j = i + 1;
          while (j < lines.length && isBlank(lines[j])) j += 1;
          if (j < lines.length && indentOf(lines[j]) > baseIndent && children.length) {
            children.push('');
            i = j;
            continue;
          }
          break;
        }
        const lm = MD.list.exec(line);
        if (lm && indentOf(lm[1]) <= baseIndent) break;
        if (indentOf(line) > baseIndent && (lm || children.length || MD.fence.test(line))) {
          children.push(line);
          i += 1;
          continue;
        }
        if (!lm && !children.length && !startsBlock(lines, i)) {
          content.push(line.trim());
          i += 1;
          continue;
        }
        break;
      }
      const li = el('li');
      li.appendChild(renderInline(content.join('\n')));
      if (children.length) {
        const minIndent = Math.min(...children.filter((l) => !isBlank(l)).map(indentOf));
        const dedented = children.map((l) => dedent(l, minIndent));
        li.appendChild(renderBlocks(dedented, depth + 1));
      }
      list.appendChild(li);
    }
    return [list, i];
  }

  function dedent(line, width) {
    let removed = 0;
    let idx = 0;
    while (idx < line.length && removed < width) {
      if (line[idx] === ' ') removed += 1;
      else if (line[idx] === '\t') removed += 4;
      else break;
      idx += 1;
    }
    return line.slice(idx);
  }

  function splitRow(line) {
    let text = line.trim();
    if (text.startsWith('|')) text = text.slice(1);
    if (text.endsWith('|') && !text.endsWith('\\|')) text = text.slice(0, -1);
    const cells = [];
    let cur = '';
    let inCode = false;
    for (let k = 0; k < text.length; k += 1) {
      const ch = text[k];
      if (ch === '\\' && text[k + 1] === '|') { cur += '|'; k += 1; continue; }
      if (ch === '`') inCode = !inCode;
      if (ch === '|' && !inCode) { cells.push(cur.trim()); cur = ''; continue; }
      cur += ch;
    }
    cells.push(cur.trim());
    return cells;
  }

  function parseTable(lines, start) {
    const header = splitRow(lines[start]);
    const aligns = splitRow(lines[start + 1]).map((cell) => {
      const left = cell.startsWith(':');
      const right = cell.endsWith(':');
      return left && right ? 'center' : right ? 'right' : left ? 'left' : null;
    });
    const table = el('table');
    const headRow = el('tr');
    header.forEach((cell, idx) => {
      const th = el('th', { scope: 'col' });
      if (aligns[idx]) th.style.textAlign = aligns[idx];
      th.appendChild(renderInline(cell));
      headRow.appendChild(th);
    });
    table.appendChild(el('thead', null, headRow));
    const tbody = el('tbody');
    let i = start + 2;
    while (i < lines.length && !isBlank(lines[i]) && lines[i].includes('|')) {
      const cells = splitRow(lines[i]);
      const tr = el('tr');
      for (let idx = 0; idx < header.length; idx += 1) {
        const td = el('td');
        if (aligns[idx]) td.style.textAlign = aligns[idx];
        td.appendChild(renderInline(cells[idx] || ''));
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
      i += 1;
    }
    table.appendChild(tbody);
    return [el('div', { class: 'table-wrap' }, table), i];
  }

  const LINK_RE = /^\[((?:\\.|[^\]\\])*)\]\(\s*<?([^\s<>()]+(?:\([^\s<>()]*\))?[^\s<>()]*)>?(?:\s+(?:"[^"]*"|'[^']*'))?\s*\)/;
  const ANGLE_URL_RE = /^<(https?:\/\/[^\s<>]+)>/i;
  const BARE_URL_RE = /^https?:\/\/[^\s<>"'`，。；：！？、（）【】《》]+/i;

  function renderInline(text, opts = { links: true }, depth = 0) {
    const frag = document.createDocumentFragment();
    let buf = '';
    const flush = () => {
      if (!buf) return;
      const parts = buf.split('\n');
      parts.forEach((part, idx) => {
        if (idx) frag.appendChild(el('br'));
        if (part) frag.appendChild(document.createTextNode(part));
      });
      buf = '';
    };
    if (depth > 8) {
      buf = text;
      flush();
      return frag;
    }
    const wrap = (tag, inner) => {
      flush();
      const node = el(tag);
      node.appendChild(renderInline(inner, opts, depth + 1));
      frag.appendChild(node);
    };

    let i = 0;
    while (i < text.length) {
      const ch = text[i];
      const next = text[i + 1];

      if (ch === '\\' && next && /[\\`*_{}[\]()#+\-.!|~<>]/.test(next)) {
        buf += next;
        i += 2;
        continue;
      }

      if (ch === '`') {
        let n = 1;
        while (text[i + n] === '`') n += 1;
        const ticks = '`'.repeat(n);
        const end = text.indexOf(ticks, i + n);
        if (end > -1) {
          flush();
          let code = text.slice(i + n, end);
          if (/^ .* $/.test(code)) code = code.slice(1, -1);
          frag.appendChild(el('code', { text: code }));
          i = end + n;
          continue;
        }
        buf += ticks;
        i += n;
        continue;
      }

      if ((ch === '*' || ch === '_') && next === ch) {
        const end = text.indexOf(ch + ch, i + 2);
        if (end > i + 2 && !/\s/.test(text[i + 2]) && !/\s/.test(text[end - 1])) {
          wrap('strong', text.slice(i + 2, end));
          i = end + 2;
          continue;
        }
        buf += ch + ch;
        i += 2;
        continue;
      }

      if (ch === '~' && next === '~') {
        const end = text.indexOf('~~', i + 2);
        if (end > i + 2) {
          wrap('del', text.slice(i + 2, end));
          i = end + 2;
          continue;
        }
      }

      if (ch === '*' && next && !/\s/.test(next)) {
        let end = -1;
        for (let j = i + 1; j < text.length; j += 1) {
          if (text[j] === '\n') break;
          if (text[j] === '*' && text[j + 1] !== '*' && !/\s/.test(text[j - 1])) { end = j; break; }
        }
        if (end > i + 1) {
          wrap('em', text.slice(i + 1, end));
          i = end + 1;
          continue;
        }
      }

      if (ch === '_' && next && !/\s/.test(next) && (i === 0 || !/[\p{L}\p{N}]/u.test(text[i - 1]))) {
        let end = -1;
        for (let j = i + 1; j < text.length; j += 1) {
          if (text[j] === '\n') break;
          if (text[j] === '_' && !/\s/.test(text[j - 1]) && (j + 1 >= text.length || !/[\p{L}\p{N}_]/u.test(text[j + 1]))) { end = j; break; }
        }
        if (end > i + 1) {
          wrap('em', text.slice(i + 1, end));
          i = end + 1;
          continue;
        }
      }

      if (ch === '[') {
        const m = LINK_RE.exec(text.slice(i));
        if (m) {
          flush();
          const label = m[1].replace(/\\(.)/g, '$1');
          const href = opts.links ? safeHttpUrl(m[2]) : null;
          if (href) {
            const a = el('a', { href, target: '_blank', rel: 'noopener noreferrer' });
            a.appendChild(renderInline(label, { links: false }, depth + 1));
            frag.appendChild(a);
          } else {
            frag.appendChild(renderInline(label, { links: false }, depth + 1));
          }
          i += m[0].length;
          continue;
        }
      }

      if (opts.links && ch === '<') {
        const m = ANGLE_URL_RE.exec(text.slice(i));
        const href = m && safeHttpUrl(m[1]);
        if (href) {
          flush();
          frag.appendChild(el('a', { href, target: '_blank', rel: 'noopener noreferrer', text: m[1] }));
          i += m[0].length;
          continue;
        }
      }

      if (opts.links && (ch === 'h' || ch === 'H') && (i === 0 || !/[\p{L}\p{N}/]/u.test(text[i - 1]))) {
        const m = BARE_URL_RE.exec(text.slice(i));
        if (m) {
          const raw = m[0].replace(/[.,;:!?)\]]+$/, '');
          const href = safeHttpUrl(raw);
          if (href) {
            flush();
            frag.appendChild(el('a', { href, target: '_blank', rel: 'noopener noreferrer', text: raw }));
            i += raw.length;
            continue;
          }
        }
      }

      buf += ch;
      i += 1;
    }
    flush();
    return frag;
  }

  // ---------------------------------------------------------------------------
  // 状态
  // ---------------------------------------------------------------------------

  const state = {
    view: null,
    route: { name: 'home', id: null },
    currentId: null,
    history: [],
    historyStatus: 'loading', // loading | ready | error
    historyError: '',
    details: new Map(),
    detailErrors: new Map(),
    detailLoading: new Set(),
    submitting: false,
    health: null,
    healthError: '',
    evidenceSelection: new Map(),
    lastQuestionHash: '#/',
    renderKeys: { answer: '', evidence: '' },
  };

  const settings = {
    snapshot: null,
    status: 'idle', // idle | loading | ready | error
    saving: false,
    testing: false,
    testController: null,
  };

  // ---------------------------------------------------------------------------
  // 路由
  // ---------------------------------------------------------------------------

  function parseRoute(hash) {
    const raw = (hash || '').replace(/^#/, '');
    const parts = raw.split('/').filter(Boolean);
    const decode = (part) => {
      try { return decodeURIComponent(part); } catch { return part; }
    };
    if (!parts.length) return { name: 'home', id: null };
    if (parts[0] === 'new') return { name: 'new', id: null };
    if (parts[0] === 'q' && parts[1]) return { name: 'question', id: decode(parts[1]) };
    if (parts[0] === 'evidence') return { name: 'evidence', id: parts[1] ? decode(parts[1]) : null };
    if (parts[0] === 'settings') return { name: 'settings', id: null };
    return { name: 'home', id: null };
  }

  const questionHash = (id) => `#/q/${encodeURIComponent(id)}`;
  const evidenceHash = (id) => (id ? `#/evidence/${encodeURIComponent(id)}` : '#/evidence');

  let lastHash = window.location.hash;
  let leaveGuardOpen = false;

  function navigate(hash) {
    if (window.location.hash === hash) applyRoute();
    else window.location.hash = hash;
  }

  async function onHashChange() {
    const target = window.location.hash;
    const next = parseRoute(target);
    if (state.view === 'settings' && next.name !== 'settings' && isSettingsDirty()) {
      // 先回到设置页地址，再询问是否放弃修改。
      window.history.replaceState(null, '', lastHash || '#/settings');
      if (leaveGuardOpen) return;
      leaveGuardOpen = true;
      const ok = await confirmDiscard();
      leaveGuardOpen = false;
      if (!ok) return;
      discardSettingsEdits();
      navigate(target);
      return;
    }
    applyRoute();
  }

  function applyRoute() {
    lastHash = window.location.hash;
    const route = parseRoute(lastHash);
    state.route = route;
    closeHistoryDrawer();

    if (route.name === 'home') {
      if (state.historyStatus === 'ready') {
        const latest = state.history[0];
        window.location.replace(latest ? questionHash(latest.id) : '#/new');
        return;
      }
      state.currentId = null;
      showView('main');
      renderAll();
      return;
    }

    if (route.name === 'new' || route.name === 'question') {
      state.currentId = route.id;
      if (route.id) state.lastQuestionHash = questionHash(route.id);
      else state.lastQuestionHash = '#/new';
      showView('main');
      if (route.id) ensureDetail(route.id);
      renderAll();
      if (route.name === 'new') focusComposer();
      return;
    }

    if (route.name === 'evidence') {
      state.currentId = route.id;
      $('evidence-back').setAttribute('href', route.id ? questionHash(route.id) : state.lastQuestionHash);
      showView('evidence');
      if (route.id) ensureDetail(route.id);
      renderAll();
      return;
    }

    if (route.name === 'settings') {
      $('settings-back').setAttribute('href', state.lastQuestionHash);
      const entering = state.view !== 'settings';
      showView('settings');
      if (entering) loadSettings();
    }
  }

  function showView(name) {
    if (state.view === name) return;
    state.view = name;
    for (const section of document.querySelectorAll('.view')) {
      section.hidden = section.dataset.view !== name;
    }
    state.renderKeys.answer = '';
    state.renderKeys.evidence = '';
    const titles = { main: '资料问答工作台', evidence: '引用来源 / 证据层 · 资料问答工作台', settings: '设置 · 资料问答工作台' };
    document.title = titles[name] || titles.main;
    window.scrollTo(0, 0);
  }

  function renderAll() {
    if (state.view === 'main') {
      renderHistory();
      renderAnswer();
      updateComposer();
    } else if (state.view === 'evidence') {
      renderEvidence();
    }
  }

  // ---------------------------------------------------------------------------
  // 数据加载与轮询
  // ---------------------------------------------------------------------------

  async function loadHistory({ quiet = false } = {}) {
    if (!quiet) {
      state.historyStatus = state.history.length ? state.historyStatus : 'loading';
      renderHistory();
    }
    try {
      const data = await api('/api/history');
      const items = Array.isArray(data.items) ? data.items.filter((item) => item && item.id !== undefined && item.id !== null) : [];
      state.history = items.map((item) => ({ ...item, id: String(item.id) }));
      state.historyStatus = 'ready';
      state.historyError = '';
      // 历史状态已变化的缓存详情需要重新获取。
      for (const item of state.history) {
        const cached = state.details.get(item.id);
        if (cached && cached.status !== item.status) {
          if (item.id === state.currentId) refreshDetail(item.id);
          else state.details.delete(item.id);
        }
      }
    } catch (err) {
      if (!state.history.length || state.historyStatus !== 'ready') {
        state.historyStatus = 'error';
        state.historyError = err.message;
      }
    }
    if (state.route.name === 'home' && state.historyStatus === 'ready') applyRoute();
    else renderAll();
  }

  function ensureDetail(id) {
    if (!id) return;
    const cached = state.details.get(id);
    if (cached && !PENDING.has(cached.status)) return;
    if (!cached) refreshDetail(id);
  }

  async function refreshDetail(id) {
    if (state.detailLoading.has(id)) return;
    state.detailLoading.add(id);
    try {
      const data = await api(`/api/questions/${encodeURIComponent(id)}`);
      const detail = { ...data, id: String(data.id ?? id) };
      state.details.set(id, detail);
      state.detailErrors.delete(id);
      syncHistoryItem(detail);
    } catch (err) {
      if (!state.details.has(id) || err.status === 404) {
        state.details.delete(id);
        state.detailErrors.set(id, err.status === 404 ? '找不到这条问题记录，它可能已被删除。' : err.message);
      }
    } finally {
      state.detailLoading.delete(id);
    }
    if (id === state.currentId) renderAll();
  }

  function syncHistoryItem(detail) {
    const item = state.history.find((h) => h.id === detail.id);
    if (!item) return;
    const changed = item.status !== detail.status || item.stage !== detail.stage || item.warning !== detail.warning;
    item.status = detail.status;
    item.stage = detail.stage;
    if (detail.warning !== undefined) item.warning = detail.warning;
    if (changed && state.view === 'main') renderHistory();
  }

  let pollTimer = null;

  function schedulePoll(delay = POLL_MS) {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(pollTick, delay);
  }

  async function pollTick() {
    try {
      if (document.hidden) return;
      const tasks = [];
      const id = state.currentId;
      const current = id && state.details.get(id);
      if (id && (state.view === 'main' || state.view === 'evidence') && current && PENDING.has(current.status)) {
        tasks.push(refreshDetail(id));
      }
      if (state.historyStatus === 'ready' && state.history.some((h) => PENDING.has(h.status))) {
        tasks.push(loadHistory({ quiet: true }));
      }
      await Promise.allSettled(tasks);
    } finally {
      schedulePoll();
    }
  }

  let healthTimer = null;

  async function checkHealth() {
    clearTimeout(healthTimer);
    const banner = $('health-banner');
    try {
      const data = await api('/api/health', { timeout: 10000 });
      state.health = { ready: Boolean(data.ready), busy: Boolean(data.busy), message: str(data.message) };
      state.healthError = '';
    } catch (err) {
      state.health = null;
      state.healthError = err.message;
    }
    const wasBlocked = banner.dataset.blocked === 'true';
    if (state.healthError) {
      banner.textContent = state.healthError;
      banner.className = 'health-banner error';
      banner.hidden = false;
      banner.dataset.blocked = 'true';
    } else if (!state.health.ready) {
      banner.textContent = state.health.message || '服务正在准备中，请稍候…';
      banner.className = 'health-banner';
      banner.hidden = false;
      banner.dataset.blocked = 'true';
    } else {
      banner.hidden = true;
      banner.dataset.blocked = 'false';
      if (wasBlocked && state.historyStatus === 'error') loadHistory();
    }
    updateComposer();
    if (banner.dataset.blocked === 'true') healthTimer = setTimeout(checkHealth, HEALTH_RETRY_MS);
  }

  // ---------------------------------------------------------------------------
  // 主界面：历史
  // ---------------------------------------------------------------------------

  function renderHistory() {
    const list = $('history-list');
    const stateBox = $('history-state');
    $('history-count').textContent = pad2(state.history.length);
    list.replaceChildren();
    stateBox.replaceChildren();
    stateBox.hidden = true;
    stateBox.className = 'side-state';

    if (state.historyStatus === 'loading' && !state.history.length) {
      stateBox.textContent = '正在加载历史问题…';
      stateBox.hidden = false;
      return;
    }
    if (state.historyStatus === 'error' && !state.history.length) {
      stateBox.className = 'side-state error';
      stateBox.append(el('p', { text: state.historyError || '历史问题加载失败' }), el('button', { type: 'button', class: 'btn small', text: '重新加载', onclick: () => loadHistory() }));
      stateBox.hidden = false;
      return;
    }
    if (!state.history.length) {
      stateBox.textContent = '还没有提问记录。在右侧输入框提出第一个问题。';
      stateBox.hidden = false;
      return;
    }

    for (const item of state.history) {
      const active = state.currentId === item.id;
      const meta = [];
      if (PENDING.has(item.status)) meta.push(el('span', { class: 'badge pending', text: str(item.stage) || STATUS_LABEL[item.status] }));
      else if (item.status && item.status !== 'answered') meta.push(el('span', { class: `badge ${item.status}`, text: STATUS_LABEL[item.status] || str(item.status) }));
      if (item.warning) meta.push(el('span', { class: 'badge warn', text: '有提示', title: str(item.warning) }));
      const button = el('button', {
        type: 'button',
        class: `history-item${active ? ' active' : ''}`,
        'aria-current': active ? 'true' : null,
        title: formatTime(item.created_at) || null,
        onclick: () => navigate(questionHash(item.id)),
      },
      el('span', { class: 'history-query', text: str(item.query) || '（空问题）' }),
      meta.length ? el('span', { class: 'history-meta' }, meta) : null);
      list.appendChild(el('li', null, button));
    }
  }

  // ---------------------------------------------------------------------------
  // 主界面：回答
  // ---------------------------------------------------------------------------

  function renderAnswer() {
    const box = $('answer');
    const id = state.currentId;
    const detail = id ? state.details.get(id) : null;
    const error = id ? state.detailErrors.get(id) : '';
    const key = JSON.stringify([
      state.route.name, id, state.historyStatus === 'loading',
      detail && [detail.status, detail.stage, detail.warning, detail.error, detail.answer_markdown],
      error,
    ]);
    if (key === state.renderKeys.answer) return;
    const idChanged = !state.renderKeys.answer || JSON.parse(state.renderKeys.answer)[1] !== id;
    state.renderKeys.answer = key;
    box.replaceChildren();

    if (!id) {
      if (state.route.name === 'home' && state.historyStatus === 'loading') {
        box.appendChild(pendingBox('正在载入…'));
      } else {
        box.appendChild(el('div', { class: 'empty' },
          el('h2', { text: '想从资料里查点什么？' }),
          el('p', { text: '提问后会检索全部本地资料，再结合资料生成回答。' }),
          el('p', { text: '每个问题独立作答，不会参考之前的问答。' })));
      }
      if (idChanged) $('answer-scroll').scrollTop = 0;
      return;
    }

    if (!detail) {
      box.appendChild(el('p', { class: 'eyebrow', text: 'AI 回答' }));
      if (error) {
        box.appendChild(el('div', { class: 'notice error', role: 'alert', text: error }));
        box.appendChild(el('button', { type: 'button', class: 'btn small', text: '重试', onclick: () => { state.detailErrors.delete(id); renderAll(); refreshDetail(id); } }));
      } else {
        box.appendChild(pendingBox('正在加载回答…'));
      }
      return;
    }

    box.appendChild(el('p', { class: 'eyebrow', text: 'AI 回答' }));
    box.appendChild(el('p', { class: 'answer-question', text: str(detail.query) }));

    const status = detail.status;
    if (PENDING.has(status)) {
      box.appendChild(pendingBox(str(detail.stage) || STATUS_LABEL[status]));
      box.appendChild(el('p', { class: 'fine', text: '可以离开此页，回答完成后会保存在历史问题中。' }));
    } else {
      if (detail.warning) box.appendChild(el('div', { class: 'notice warn', role: 'note', text: str(detail.warning) }));
      const markdown = str(detail.answer_markdown).trim();
      if (status === 'failed' || status === 'interrupted') {
        const prefix = status === 'failed' ? '回答生成失败' : '回答已中断';
        box.appendChild(el('div', { class: 'notice error', role: 'alert', text: detail.error ? `${prefix}：${detail.error}` : `${prefix}，可以重新提问。` }));
        box.appendChild(el('button', { type: 'button', class: 'btn small', onclick: () => reuseQuestion(detail.query) }, icon(ICON.retry), '重新提问'));
      } else if (status === 'no_answer') {
        box.appendChild(el('div', { class: 'notice info', text: '资料中没有找到足以回答这个问题的内容。' }));
        if (markdown) box.appendChild(el('div', { class: 'md' }, renderMarkdown(markdown)));
      } else if (markdown) {
        box.appendChild(el('div', { class: 'md' }, renderMarkdown(markdown)));
      } else {
        box.appendChild(el('div', { class: 'notice info', text: '这次没有返回回答内容。' }));
      }
    }
    if (idChanged) $('answer-scroll').scrollTop = 0;
  }

  function pendingBox(text) {
    return el('div', { class: 'pending-box', role: 'status' }, el('span', { class: 'spinner', 'aria-hidden': 'true' }), el('span', { text }));
  }

  function reuseQuestion(query) {
    const input = $('question-input');
    input.value = str(query);
    saveDraft();
    autosize();
    navigate('#/new');
    focusComposer();
  }

  // ---------------------------------------------------------------------------
  // 输入框
  // ---------------------------------------------------------------------------

  function saveDraft() {
    try { window.sessionStorage.setItem(DRAFT_KEY, $('question-input').value); } catch { /* 存储不可用时仅保留在内存 */ }
  }

  function restoreDraft() {
    try {
      const draft = window.sessionStorage.getItem(DRAFT_KEY);
      if (draft) $('question-input').value = draft;
    } catch { /* ignore */ }
  }

  function autosize() {
    const input = $('question-input');
    input.style.height = 'auto';
    input.style.height = `${Math.max(input.scrollHeight, 54)}px`;
  }

  function focusComposer() {
    const input = $('question-input');
    if (state.view === 'main' && !window.matchMedia('(max-width: 760px)').matches) {
      input.focus({ preventScroll: true });
    }
  }

  let composerError = '';
  let pendingSubmission = null;
  try { pendingSubmission = JSON.parse(window.sessionStorage.getItem(SUBMISSION_KEY) || 'null'); } catch { /* Optional persistence. */ }

  function submissionKey(query) {
    if (!pendingSubmission || pendingSubmission.query !== query || !/^[0-9a-f]{32}$/.test(pendingSubmission.key)) {
      const bytes = crypto.getRandomValues(new Uint8Array(16));
      const key = [...bytes].map(value => value.toString(16).padStart(2, '0')).join('');
      pendingSubmission = { query, key };
      try { window.sessionStorage.setItem(SUBMISSION_KEY, JSON.stringify(pendingSubmission)); } catch { /* Keep in memory. */ }
    }
    return pendingSubmission.key;
  }

  function clearSubmission() {
    pendingSubmission = null;
    try { window.sessionStorage.removeItem(SUBMISSION_KEY); } catch { /* Optional persistence. */ }
  }

  function updateComposer() {
    const button = $('send-btn');
    const hint = $('composer-hint');
    const blocked = Boolean(state.healthError) || (state.health && !state.health.ready);
    button.disabled = state.submitting || blocked;
    button.classList.toggle('busy', state.submitting);
    button.setAttribute('aria-label', state.submitting ? '正在提交' : '发送问题');
    $('question-input').setAttribute('aria-busy', state.submitting ? 'true' : 'false');
    if (composerError) {
      hint.textContent = composerError;
      hint.className = 'composer-hint error';
    } else if (blocked) {
      hint.textContent = state.healthError ? '暂时无法连接服务' : (state.health.message || '服务准备中，暂时不能提问');
      hint.className = 'composer-hint';
    } else {
      hint.textContent = '默认检索全部资料 · Enter 发送，Shift+Enter 换行';
      hint.className = 'composer-hint';
    }
  }

  async function submitQuestion() {
    const input = $('question-input');
    const query = input.value.trim();
    if (state.submitting) return;
    if (!query) {
      input.focus();
      return;
    }
    if (state.healthError || (state.health && !state.health.ready)) {
      checkHealth();
      return;
    }
    state.submitting = true;
    composerError = '';
    updateComposer();
    try {
      const data = await api('/api/questions', { method: 'POST', body: { query }, timeout: 30000,
        headers: { 'Idempotency-Key': submissionKey(query) } });
      clearSubmission();
      const detail = { ...data, id: String(data.id) };
      state.details.set(detail.id, detail);
      state.detailErrors.delete(detail.id);
      state.history = [
        { id: detail.id, query: detail.query, status: detail.status, stage: detail.stage, created_at: detail.created_at, warning: detail.warning },
        ...state.history.filter((h) => h.id !== detail.id),
      ];
      state.historyStatus = 'ready';
      if (input.value.trim() === query) input.value = '';
      saveDraft();
      autosize();
      navigate(questionHash(detail.id));
      loadHistory({ quiet: true });
      schedulePoll();
    } catch (err) {
      composerError = err.message || '提交失败，请稍后重试';
      if (err.status === 0) checkHealth();
    } finally {
      state.submitting = false;
      updateComposer();
    }
  }

  // ---------------------------------------------------------------------------
  // 证据层
  // ---------------------------------------------------------------------------

  function evidenceTitle(item) {
    const isFigure = item.type === 'figure';
    const primary = isFigure ? (str(item.label) || str(item.section)) : (str(item.section) || str(item.label));
    return primary || (isFigure ? '图表证据' : '正文摘录');
  }

  function pageText(page) {
    if (page === null || page === undefined || page === '') return '';
    return `第 ${page} 页`;
  }

  function evidenceMeta(item) {
    const title = evidenceTitle(item);
    return [str(item.label), str(item.section)].filter((part) => part && part !== title).concat(pageText(item.page) || []).filter(Boolean);
  }

  function groupEvidence(items) {
    const groups = new Map();
    for (const item of items) {
      const key = str(item.document_id) || str(item.title) || '未知资料';
      if (!groups.has(key)) groups.set(key, { documentId: str(item.document_id), title: str(item.title), items: [] });
      groups.get(key).items.push(item);
    }
    return [...groups.values()];
  }

  function renderEvidence() {
    const id = state.currentId;
    const nav = $('evidence-nav');
    const detailBox = $('evidence-detail');
    const summary = $('evidence-summary');
    const questionEl = $('evidence-question');
    const detail = id ? state.details.get(id) : null;
    const error = id ? state.detailErrors.get(id) : '';
    const selected = id ? state.evidenceSelection.get(id) : null;
    const key = JSON.stringify([id, error, selected, detail && [detail.status, detail.stage, detail.query, detail.evidence]]);
    if (key === state.renderKeys.evidence) return;
    state.renderKeys.evidence = key;

    nav.replaceChildren();
    detailBox.replaceChildren();
    summary.textContent = '';

    if (!id) {
      questionEl.textContent = '尚未选择问题';
      detailBox.appendChild(emptyBlock('还没有可查看的证据', '先在问答页提出或选择一个问题，再来查看它引用的资料。', '返回问答', state.lastQuestionHash));
      return;
    }
    if (!detail) {
      questionEl.textContent = state.history.find((h) => h.id === id)?.query || '';
      if (error) {
        detailBox.appendChild(el('div', { class: 'notice error', role: 'alert', text: error }));
        detailBox.appendChild(el('button', { type: 'button', class: 'btn small', text: '重试', onclick: () => { state.detailErrors.delete(id); state.renderKeys.evidence = ''; renderAll(); refreshDetail(id); } }));
      } else {
        detailBox.appendChild(pendingBox('正在加载证据…'));
      }
      return;
    }

    questionEl.textContent = str(detail.query);

    if (PENDING.has(detail.status)) {
      summary.textContent = '本次回答 · 生成中';
      detailBox.appendChild(pendingBox(str(detail.stage) || STATUS_LABEL[detail.status]));
      detailBox.appendChild(el('p', { class: 'fine', text: '回答完成后，这里会列出它实际引用的资料。' }));
      return;
    }

    const items = (Array.isArray(detail.evidence) ? detail.evidence : []).filter((item) => item && typeof item === 'object');
    if (!items.length) {
      summary.textContent = '本次回答 · 0 条证据';
      const message = detail.status === 'failed' || detail.status === 'interrupted'
        ? '这次回答没有完成，没有可查看的引用证据。'
        : '这次回答没有引用资料中的具体证据。';
      detailBox.appendChild(emptyBlock('没有引用证据', message, '返回问答', questionHash(id)));
      return;
    }

    const groups = groupEvidence(items);
    summary.textContent = `本次回答 · ${groups.length} 份资料 · ${items.length} 条证据`;
    const current = items.find((item) => str(item.citation_id) === selected) || items[0];

    for (const group of groups) {
      const section = el('div', { class: 'source-group', role: 'group', 'aria-label': group.title || group.documentId });
      section.appendChild(el('div', { class: 'source-group-head' },
        group.documentId ? el('p', { class: 'source-kicker', text: group.documentId }) : null,
        el('p', { class: 'source-name', text: group.title || group.documentId || '未命名资料' })));
      for (const item of group.items) {
        const active = item === current;
        const citation = str(item.citation_id);
        section.appendChild(el('button', {
          type: 'button',
          class: `ev-item${active ? ' active' : ''}`,
          'aria-pressed': active ? 'true' : 'false',
          onclick: () => {
            state.evidenceSelection.set(id, citation);
            renderEvidence();
            if (window.matchMedia('(max-width: 760px)').matches) detailBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
          },
        },
        el('span', { class: 'ev-top' },
          el('span', { class: 'ev-type', text: item.type === 'figure' ? '图表' : '正文' }),
          el('span', { class: 'ev-id', text: citation })),
        el('span', { class: 'ev-title', text: evidenceTitle(item) }),
        evidenceMeta(item).length ? el('span', { class: 'ev-meta', text: evidenceMeta(item).join('  ·  ') }) : null));
      }
      nav.appendChild(section);
    }

    detailBox.appendChild(renderEvidenceDetail(current));
  }

  function emptyBlock(title, text, actionText, href) {
    return el('div', { class: 'empty' },
      el('h2', { text: title }),
      el('p', { text }),
      actionText ? el('a', { class: 'btn', href, text: actionText }) : null);
  }

  function renderEvidenceDetail(item) {
    const frag = document.createDocumentFragment();
    const title = evidenceTitle(item);
    const isFigure = item.type === 'figure';
    const loc = [str(item.title), str(item.label), str(item.section), pageText(item.page)].filter(Boolean);
    const uniqueLoc = loc.filter((part, idx) => loc.indexOf(part) === idx && part !== title);

    const actions = el('div', { class: 'ev-actions' });
    const sourceUrl = safeHttpUrl(item.source_url);
    if (sourceUrl) {
      actions.appendChild(el('a', { class: 'btn', href: sourceUrl, target: '_blank', rel: 'noopener noreferrer', title: sourceUrl }, '打开来源', icon(ICON.external)));
    }
    if (item.document_id !== undefined && item.document_id !== null && item.document_id !== '') {
      actions.appendChild(el('button', { type: 'button', class: 'btn', onclick: () => openDocument(str(item.document_id), { fromList: false }) }, '本地原文', icon(ICON.doc)));
    }

    frag.appendChild(el('div', { class: 'ev-head' },
      el('div', null,
        el('h3', { text: title }),
        uniqueLoc.length ? el('p', { class: 'ev-loc', text: uniqueLoc.join('   /   ') }) : null,
        el('p', { class: 'ev-loc', text: `引用编号 ${str(item.citation_id)}` })),
      actions.childNodes.length ? actions : null));

    if (str(item.related_answer).trim()) {
      frag.appendChild(el('div', { class: 'related' },
        el('p', { class: 'label', text: '对应回答中的内容' }),
        el('p', { class: 'quote', tabindex: 0, 'aria-label': '对应回答中的内容，可滚动查看', text: `“${str(item.related_answer).trim()}”` })));
    }

    if (isFigure) {
      const src = sameOriginUrl(item.image_url);
      const alt = [str(item.label), title].filter(Boolean)[0] || '原始图表';
      const head = el('div', { class: 'block-head' }, el('h4', { text: '原始图表' }));
      const frame = el('div', { class: 'figure-frame' });
      if (src) {
        head.appendChild(el('button', { type: 'button', class: 'zoom-btn', onclick: () => openImage(src, alt) }, '查看大图', icon(ICON.zoom)));
        const img = el('img', { src, alt, loading: 'lazy', decoding: 'async' });
        img.addEventListener('error', () => {
          frame.replaceChildren(el('p', { class: 'figure-missing', text: '图片加载失败。' }));
          head.querySelector('.zoom-btn')?.remove();
        }, { once: true });
        frame.appendChild(el('button', { type: 'button', 'aria-label': `放大查看：${alt}`, onclick: () => openImage(src, alt) }, img));
      } else {
        frame.appendChild(el('p', { class: 'figure-missing', text: '这条图表证据没有可显示的原图。' }));
      }
      frag.appendChild(head);
      frag.appendChild(frame);
      frag.appendChild(el('div', { class: 'block-head' }, el('h4', { text: '图表说明' })));
      frag.appendChild(el('p', { class: 'excerpt', text: str(item.text) || '（无图注或说明文字）' }));
      frag.appendChild(el('p', { class: 'fine', text: `原图与说明来自资料${item.title ? `《${str(item.title)}》` : ''}${item.label ? ` ${str(item.label)}` : ''}。` }));
    } else {
      frag.appendChild(el('div', { class: 'block-head' }, el('h4', { text: '原文摘录' })));
      frag.appendChild(el('p', { class: 'excerpt boxed', text: str(item.text) || '（无摘录文字）' }));
    }
    return frag;
  }

  // ---------------------------------------------------------------------------
  // 弹窗
  // ---------------------------------------------------------------------------

  function setupDialog(dialog) {
    dialog.addEventListener('click', (event) => {
      if (event.target.closest('[data-close]')) {
        dialog.close();
        return;
      }
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      const inside = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
      if (!inside) dialog.close();
    });
  }

  function openModal(dialog) {
    if (!dialog.open) dialog.showModal();
  }

  function openImage(src, alt) {
    const img = $('image-full');
    img.src = src;
    img.alt = alt;
    $('image-title').textContent = alt || '原始图表';
    openModal($('image-dialog'));
  }

  let docsToken = 0;
  let documentsCache = null;

  async function openDocsList() {
    const token = ++docsToken;
    const body = $('docs-body');
    $('docs-title').textContent = '本地资料库';
    $('docs-sub').textContent = '只读查看。当前版本不支持在界面中添加或删除资料。';
    $('docs-back-btn').hidden = true;
    body.replaceChildren(pendingBox('正在加载资料列表…'));
    openModal($('docs-dialog'));
    try {
      if (!documentsCache) {
        const data = await api('/api/documents');
        documentsCache = Array.isArray(data.documents) ? data.documents.filter((d) => d && d.id !== undefined && d.id !== null) : [];
      }
      if (token !== docsToken) return;
      body.replaceChildren();
      if (!documentsCache.length) {
        body.appendChild(el('p', { class: 'side-state', text: '资料库中还没有资料。' }));
        return;
      }
      $('docs-sub').textContent = `共 ${documentsCache.length} 份资料 · 只读查看，当前版本不支持在界面中添加或删除资料。`;
      const list = el('ul', { class: 'doc-list' });
      for (const doc of documentsCache) {
        list.appendChild(el('li', null, el('button', {
          type: 'button',
          class: 'doc-row',
          onclick: () => openDocument(String(doc.id), { fromList: true }),
        },
        el('span', { class: 'doc-icon', 'aria-hidden': 'true' }, icon(ICON.doc)),
        el('span', { class: 'row-text' }, el('strong', { text: str(doc.title) || String(doc.id) }), el('span', { text: String(doc.id) })),
        icon(ICON.chev, 'chev'))));
      }
      body.appendChild(list);
    } catch (err) {
      if (token !== docsToken) return;
      body.replaceChildren(
        el('div', { class: 'notice error', role: 'alert', text: err.message }),
        el('button', { type: 'button', class: 'btn small', text: '重新加载', onclick: openDocsList }));
    }
  }

  async function openDocument(id, { fromList }) {
    const token = ++docsToken;
    const body = $('docs-body');
    $('docs-back-btn').hidden = !fromList;
    $('docs-title').textContent = '资料原文';
    $('docs-sub').textContent = id;
    body.replaceChildren(pendingBox('正在加载原文…'));
    openModal($('docs-dialog'));
    try {
      const doc = await api(`/api/documents/${encodeURIComponent(id)}`);
      if (token !== docsToken) return;
      $('docs-title').textContent = str(doc.title) || id;
      $('docs-sub').textContent = `${str(doc.id) || id} · 原始 Markdown，只读`;
      body.replaceChildren();
      const url = safeHttpUrl(doc.source_url);
      if (url) {
        body.appendChild(el('div', { class: 'doc-links' },
          el('a', { class: 'btn small', href: url, target: '_blank', rel: 'noopener noreferrer', title: url }, '打开来源链接', icon(ICON.external))));
      }
      body.appendChild(el('pre', { class: 'doc-text', text: str(doc.text) || '（原文为空）' }));
      body.scrollTop = 0;
    } catch (err) {
      if (token !== docsToken) return;
      body.replaceChildren(
        el('div', { class: 'notice error', role: 'alert', text: err.status === 404 ? '找不到这份资料。' : err.message }),
        el('button', { type: 'button', class: 'btn small', text: '重试', onclick: () => openDocument(id, { fromList }) }));
    }
  }

  function confirmDiscard() {
    const dialog = $('confirm-dialog');
    return new Promise((resolve) => {
      dialog.returnValue = '';
      dialog.addEventListener('close', () => resolve(dialog.returnValue === 'ok'), { once: true });
      openModal(dialog);
    });
  }

  // ---------------------------------------------------------------------------
  // 设置
  // ---------------------------------------------------------------------------

  const form = () => $('settings-form');

  function readForm() {
    const f = form();
    return {
      provider: f.elements.provider.value,
      base_url: f.elements.base_url.value.trim(),
      model: f.elements.model.value.trim(),
      language: f.elements.language.value,
      detail: (f.querySelector('input[name="detail"]:checked') || {}).value || 'balanced',
    };
  }

  function comparable(values) {
    return JSON.stringify([values.provider, values.base_url, values.model, values.language, values.detail]);
  }

  function isSettingsDirty() {
    if (settings.status !== 'ready' || !settings.snapshot) return false;
    return comparable(readForm()) !== comparable(settings.snapshot) || $('set-api-key').value.length > 0;
  }

  function fillForm(data) {
    const f = form();
    const provider = f.elements.provider;
    const providerValue = str(data.provider) || 'openai-compatible';
    if (![...provider.options].some((o) => o.value === providerValue)) {
      provider.appendChild(el('option', { value: providerValue, text: `${providerValue}（不支持）` }));
    }
    provider.value = providerValue;
    f.elements.model.value = str(data.model);
    f.elements.base_url.value = str(data.base_url);
    f.elements.language.value = LANGUAGES.includes(data.language) ? data.language : 'auto';
    const detail = DETAILS.includes(data.detail) ? data.detail : 'balanced';
    for (const radio of f.querySelectorAll('input[name="detail"]')) radio.checked = radio.value === detail;

    const key = $('set-api-key');
    key.value = '';
    key.placeholder = data.key_configured ? '已配置（留空则保持不变）' : '输入 API Key';
    $('key-help').textContent = data.key_configured ? '已保存的密钥不会显示；输入新密钥将替换它。' : '密钥输入后以隐藏字符显示。';
    for (const input of f.querySelectorAll('[aria-invalid]')) input.removeAttribute('aria-invalid');
  }

  function snapshotFrom(data) {
    return {
      provider: str(data.provider) || 'openai-compatible',
      base_url: str(data.base_url).trim(),
      model: str(data.model).trim(),
      language: LANGUAGES.includes(data.language) ? data.language : 'auto',
      detail: DETAILS.includes(data.detail) ? data.detail : 'balanced',
      key_configured: Boolean(data.key_configured),
    };
  }

  function setFeedback(text, kind) {
    const node = $('save-feedback');
    node.textContent = text;
    node.className = `save-feedback${kind ? ` ${kind}` : ''}`;
  }

  function setTestStatus(stateName, text) {
    const node = $('test-status');
    node.dataset.state = stateName;
    node.querySelector('.test-text').textContent = text;
  }

  function setFormEnabled(enabled) {
    for (const control of form().querySelectorAll('input, select, button')) {
      if (control.id === 'cancel-btn' || control.id === 'open-docs-btn') continue;
      control.disabled = !enabled;
    }
    if (enabled) updateSettingsButtons();
  }

  function updateSettingsButtons() {
    const ready = settings.status === 'ready';
    $('save-btn').disabled = !ready || settings.saving || settings.testing;
    $('save-btn').textContent = settings.saving ? '正在保存…' : '保存设置';
    $('test-btn').disabled = !ready || settings.testing || settings.saving;
    $('test-btn').textContent = settings.testing ? '测试中…' : '测试连接';
  }

  async function loadSettings() {
    settings.status = 'loading';
    const stateBox = $('settings-state');
    stateBox.hidden = false;
    stateBox.className = 'side-state';
    stateBox.textContent = '正在读取设置…';
    setFormEnabled(false);
    updateSettingsButtons();
    setFeedback('保存后，从下次提问开始生效。', '');
    setTestStatus('idle', '尚未测试连接');
    try {
      const data = await api('/api/settings');
      settings.snapshot = snapshotFrom(data);
      fillForm(settings.snapshot);
      settings.status = 'ready';
      stateBox.hidden = true;
      setFormEnabled(true);
    } catch (err) {
      settings.status = 'error';
      stateBox.className = 'side-state error';
      stateBox.replaceChildren(el('p', { text: `设置读取失败：${err.message}` }), el('button', { type: 'button', class: 'btn small', text: '重新读取', onclick: loadSettings }));
      setFormEnabled(false);
    }
    updateSettingsButtons();
    observeSettingsSections();
  }

  function validateSettings(values) {
    const f = form();
    let firstInvalid = null;
    const mark = (input, bad) => {
      if (bad) {
        input.setAttribute('aria-invalid', 'true');
        firstInvalid = firstInvalid || input;
      } else {
        input.removeAttribute('aria-invalid');
      }
    };
    mark(f.elements.model, !values.model);
    const url = safeHttpUrl(values.base_url);
    mark(f.elements.base_url, !url || new URL(url).protocol !== 'https:');
    if (firstInvalid) {
      firstInvalid.focus();
      return firstInvalid === f.elements.model ? '请填写模型名称。' : '请填写有效的 HTTPS 接口地址。';
    }
    if (values.provider !== 'openai-compatible') return '当前只支持 OpenAI 兼容接口。';
    return '';
  }

  function settingsPayload() {
    const values = readForm();
    const payload = { ...values };
    const key = $('set-api-key').value.trim();
    if (key) payload.api_key = key;
    return { values, payload, key };
  }

  /** 防御性处理：即使服务端消息意外包含密钥，也不在界面上显示。 */
  function scrubSecret(message, secret) {
    let text = str(message);
    if (secret && secret.length >= 4) text = text.split(secret).join('••••');
    return text;
  }

  async function saveSettings() {
    if (settings.status !== 'ready' || settings.saving || settings.testing) return;
    const { values, payload, key } = settingsPayload();
    const invalid = validateSettings(values);
    if (invalid) {
      setFeedback(invalid, 'error');
      return;
    }
    settings.saving = true;
    setFormEnabled(false);
    updateSettingsButtons();
    setFeedback('正在保存…', '');
    try {
      const data = await api('/api/settings', { method: 'PUT', body: payload, timeout: 20000 });
      settings.snapshot = snapshotFrom(data);
      fillForm(settings.snapshot);
      setFeedback('已保存，从下次提问开始生效。', 'ok');
    } catch (err) {
      setFeedback(`保存失败：${scrubSecret(err.message, key)}`, 'error');
    } finally {
      settings.saving = false;
      setFormEnabled(true);
      updateSettingsButtons();
    }
  }

  async function testConnection() {
    if (settings.status !== 'ready' || settings.testing || settings.saving) return;
    const { values, payload, key } = settingsPayload();
    const invalid = validateSettings(values);
    if (invalid) {
      setTestStatus('fail', invalid);
      return;
    }
    settings.testing = true;
    settings.testController = new AbortController();
    updateSettingsButtons();
    setTestStatus('testing', '正在测试连接，最长可能需要 45 秒…');
    try {
      const data = await api('/api/settings/test', { method: 'POST', body: payload, timeout: TEST_TIMEOUT_MS, signal: settings.testController.signal });
      const message = scrubSecret(data.message, key);
      if (data.ok) setTestStatus('ok', message || '连接成功');
      else setTestStatus('fail', message || '连接失败');
    } catch (err) {
      if (!(settings.testController && settings.testController.signal.aborted)) {
        setTestStatus('fail', `测试失败：${scrubSecret(err.message, key)}`);
      }
    } finally {
      settings.testing = false;
      settings.testController = null;
      updateSettingsButtons();
    }
  }

  function discardSettingsEdits() {
    if (settings.testController) settings.testController.abort();
    if (settings.snapshot) fillForm(settings.snapshot);
    $('set-api-key').value = '';
    setFeedback('保存后，从下次提问开始生效。', '');
    setTestStatus('idle', '尚未测试连接');
  }

  function onSettingsInput(event) {
    if (settings.status !== 'ready') return;
    if (event.target.getAttribute('aria-invalid')) event.target.removeAttribute('aria-invalid');
    if (isSettingsDirty()) setFeedback('有未保存的修改。保存后，从下次提问开始生效。', 'dirty');
    else setFeedback('保存后，从下次提问开始生效。', '');
    const status = $('test-status').dataset.state;
    if ((status === 'ok' || status === 'fail') && event.target.name !== 'language' && event.target.name !== 'detail') {
      setTestStatus('idle', '接口配置已修改，尚未重新测试');
    }
  }

  let sectionObserver = null;

  function observeSettingsSections() {
    if (sectionObserver || !('IntersectionObserver' in window)) return;
    const links = [...document.querySelectorAll('#settings-nav a')];
    sectionObserver = new IntersectionObserver((entries) => {
      const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (!visible) return;
      for (const link of links) link.classList.toggle('active', link.dataset.section === visible.target.id);
    }, { rootMargin: '0px 0px -55% 0px', threshold: 0 });
    for (const link of links) sectionObserver.observe($(link.dataset.section));
  }

  // ---------------------------------------------------------------------------
  // 移动端历史抽屉
  // ---------------------------------------------------------------------------

  function openHistoryDrawer() {
    $('history-sidebar').classList.add('open');
    $('history-scrim').hidden = false;
    $('toggle-history-btn').setAttribute('aria-expanded', 'true');
  }

  function closeHistoryDrawer() {
    $('history-sidebar').classList.remove('open');
    $('history-scrim').hidden = true;
    $('toggle-history-btn').setAttribute('aria-expanded', 'false');
  }

  // ---------------------------------------------------------------------------
  // 事件绑定
  // ---------------------------------------------------------------------------

  function bind() {
    const input = $('question-input');
    input.addEventListener('input', () => {
      saveDraft();
      autosize();
      if (composerError) {
        composerError = '';
        updateComposer();
      }
    });
    input.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' || event.shiftKey || event.isComposing || event.keyCode === 229) return;
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      event.preventDefault();
      submitQuestion();
    });
    $('composer').addEventListener('submit', (event) => {
      event.preventDefault();
      submitQuestion();
    });

    $('new-question-btn').addEventListener('click', () => navigate('#/new'));
    $('open-evidence-btn').addEventListener('click', () => navigate(evidenceHash(state.currentId)));
    $('open-settings-btn').addEventListener('click', () => navigate('#/settings'));
    $('toggle-history-btn').addEventListener('click', () => {
      if ($('history-sidebar').classList.contains('open')) closeHistoryDrawer();
      else openHistoryDrawer();
    });
    $('history-scrim').addEventListener('click', closeHistoryDrawer);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && $('history-sidebar').classList.contains('open')) closeHistoryDrawer();
    });

    const settingsForm = form();
    settingsForm.addEventListener('submit', (event) => {
      event.preventDefault();
      saveSettings();
    });
    settingsForm.addEventListener('input', onSettingsInput);
    settingsForm.addEventListener('change', onSettingsInput);
    $('test-btn').addEventListener('click', testConnection);
    $('open-docs-btn').addEventListener('click', openDocsList);
    $('docs-back-btn').addEventListener('click', openDocsList);
    $('cancel-btn').addEventListener('click', async () => {
      if (isSettingsDirty()) {
        const ok = await confirmDiscard();
        if (!ok) return;
        discardSettingsEdits();
      }
      navigate(state.lastQuestionHash);
    });
    for (const link of document.querySelectorAll('#settings-nav a')) {
      link.addEventListener('click', (event) => {
        event.preventDefault();
        const target = $(link.dataset.section);
        const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        target.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
        for (const other of document.querySelectorAll('#settings-nav a')) other.classList.toggle('active', other === link);
      });
    }

    for (const dialog of document.querySelectorAll('dialog')) setupDialog(dialog);
    $('image-dialog').addEventListener('close', () => { $('image-full').removeAttribute('src'); });
    $('docs-dialog').addEventListener('close', () => { docsToken += 1; });

    window.addEventListener('hashchange', onHashChange);
    window.addEventListener('beforeunload', (event) => {
      if (state.view === 'settings' && isSettingsDirty()) {
        event.preventDefault();
        event.returnValue = '';
      }
    });
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) {
        schedulePoll(0);
        if (state.healthError || (state.health && !state.health.ready)) checkHealth();
      }
    });
  }

  function init() {
    bind();
    restoreDraft();
    autosize();
    applyRoute();
    checkHealth();
    loadHistory();
    schedulePoll();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
