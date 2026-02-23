const { Plugin, ItemView, Setting, MarkdownRenderer } = require('obsidian');

const VIEW_TYPE = 'obsidian-assistant-view';

const DEFAULT_SETTINGS = {
  serverUrl: 'http://127.0.0.1:8000',
  autoOpen: true,
  saveChatsToVault: true,
  chatsFolder: 'z-Bev Chats',
  maxHistoryMessages: 8
};

class AssistantView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
  }

  getViewType() {
    return VIEW_TYPE;
  }

  getDisplayText() {
    return 'Obsidian Assistant';
  }

  async onOpen() {
    this.render();
  }

  render() {
    const contentEl = this.contentEl;
    contentEl.empty();
    contentEl.addClass('obsidian-assistant-view');

    const container = contentEl.createDiv({ cls: 'assistant-container' });
    if (this.plugin.manifest && this.plugin.manifest.dir) {
      const spritePath = `${this.plugin.manifest.dir}/assets/bev-poses.png`;
      const spriteUrl = this.app.vault.adapter.getResourcePath(spritePath);
      container.style.setProperty('--bev-sprite', `url("${spriteUrl}")`);
    }
    const header = container.createDiv({ cls: 'assistant-header' });
    const headerTop = header.createDiv({ cls: 'assistant-header-top' });
    headerTop.createDiv({ text: 'Bev', cls: 'assistant-title' });
    const statusEl = headerTop.createDiv({ text: 'Connecting...', cls: 'assistant-status' });
    statusEl.dataset.state = 'connecting';
    const settingsToggle = headerTop.createEl('button', { cls: 'assistant-settings-toggle', text: 'Settings' });

    const headerBottom = header.createDiv({ cls: 'assistant-header-bottom' });
    const modeGroup = headerBottom.createDiv({ cls: 'assistant-mode-group' });
    modeGroup.createDiv({ text: 'Mode', cls: 'assistant-mode-label' });
    const modeButtons = modeGroup.createDiv({ cls: 'assistant-mode-buttons' });
    const modeBtnSafe = modeButtons.createEl('button', { cls: 'assistant-mode-btn', text: 'Safe' });
    const modeBtnPower = modeButtons.createEl('button', { cls: 'assistant-mode-btn', text: 'Power' });
    const modeBtnFull = modeButtons.createEl('button', { cls: 'assistant-mode-btn', text: 'Full' });
    modeBtnSafe.dataset.mode = 'safe';
    modeBtnPower.dataset.mode = 'power';
    modeBtnFull.dataset.mode = 'full';
    modeBtnSafe.setAttr('title', 'Read/search + edit with approval only.');
    modeBtnPower.setAttr('title', 'Safe mode + terminal/python (ask first).');
    modeBtnFull.setAttr('title', 'Power mode + self-improve (ask first).');

    const sessionBar = headerBottom.createDiv({ cls: 'assistant-session' });
    sessionBar.createDiv({ text: 'Chat', cls: 'assistant-session-label' });
    const sessionSelect = sessionBar.createEl('select', { cls: 'assistant-session-select' });
    const newSessionBtn = sessionBar.createEl('button', { cls: 'assistant-session-new', text: 'New chat' });

    const chat = container.createDiv({ cls: 'assistant-chat' });
    const form = container.createEl('form', { cls: 'assistant-form' });

    const input = form.createEl('textarea', {
      cls: 'assistant-input',
      placeholder: 'Ask about your vault or papers...'
    });

    const toolbar = form.createDiv({ cls: 'assistant-toolbar' });
    const editLabel = toolbar.createEl('label', { cls: 'assistant-toggle' });
    const editToggle = editLabel.createEl('input', { type: 'checkbox' });
    editLabel.appendText(' Edit');
    editLabel.setAttr('title', 'Propose edits to the selected note (requires approval).');

    const webLabel = toolbar.createEl('label', { cls: 'assistant-toggle' });
    const webToggle = webLabel.createEl('input', { type: 'checkbox' });
    webLabel.appendText(' Web');
    webLabel.setAttr('title', 'Allow web search when answering.');

    const activeRow = form.createDiv({ cls: 'assistant-active-note' });
    activeRow.createDiv({ cls: 'assistant-active-label', text: 'Active note' });
    const activeName = activeRow.createDiv({ cls: 'assistant-active-name', text: '(none)' });

    const fileRow = form.createDiv({ cls: 'assistant-file-row is-hidden' });
    fileRow.createDiv({ cls: 'assistant-file-label', text: 'Editing:' });
    const fileName = fileRow.createDiv({ cls: 'assistant-file-name', text: '(select a note)' });
    const changeFileBtn = fileRow.createEl('button', { cls: 'assistant-file-change', text: 'Change', type: 'button' });
    const pathWrap = fileRow.createDiv({ cls: 'assistant-path-wrap is-hidden' });
    const pathInput = pathWrap.createEl('input', {
      cls: 'assistant-path',
      type: 'text',
      placeholder: 'Note path (auto-filled)'
    });

    const settingsPanel = form.createDiv({ cls: 'assistant-settings is-hidden' });

    const toolsRow = settingsPanel.createDiv({ cls: 'assistant-settings-row' });
    toolsRow.createDiv({ text: 'Tools', cls: 'assistant-settings-label' });
    const toolsWrap = toolsRow.createDiv({ cls: 'assistant-settings-toggles' });
    const readOnlyLabel = toolsWrap.createEl('label', { cls: 'assistant-toggle' });
    const readOnlyToggle = readOnlyLabel.createEl('input', { type: 'checkbox' });
    readOnlyLabel.appendText(' Read only');

    const allowEditLabel = toolsWrap.createEl('label', { cls: 'assistant-toggle' });
    const allowEditToggle = allowEditLabel.createEl('input', { type: 'checkbox' });
    allowEditLabel.appendText(' Edit with approval');

    const terminalLabel = toolsWrap.createEl('label', { cls: 'assistant-toggle' });
    const terminalToggle = terminalLabel.createEl('input', { type: 'checkbox' });
    terminalLabel.appendText(' Terminal');

    const pythonLabel = toolsWrap.createEl('label', { cls: 'assistant-toggle' });
    const pythonToggle = pythonLabel.createEl('input', { type: 'checkbox' });
    pythonLabel.appendText(' Python');

    const selfImproveLabel = toolsWrap.createEl('label', { cls: 'assistant-toggle' });
    const selfImproveToggle = selfImproveLabel.createEl('input', { type: 'checkbox' });
    selfImproveLabel.appendText(' Self-improve');

    const responseRow = settingsPanel.createDiv({ cls: 'assistant-settings-row' });
    responseRow.createDiv({ text: 'Response', cls: 'assistant-settings-label' });
    const responseSelect = responseRow.createEl('select', { cls: 'assistant-select' });
    [
      { value: 128, text: 'Short (128)' },
      { value: 256, text: 'Medium (256)' },
      { value: 512, text: 'Long (512)' },
      { value: 1024, text: 'Very long (1024)' }
    ].forEach((opt) => {
      const option = responseSelect.createEl('option');
      option.value = String(opt.value);
      option.text = opt.text;
    });

    const contextRow = settingsPanel.createDiv({ cls: 'assistant-settings-row' });
    contextRow.createDiv({ text: 'Context', cls: 'assistant-settings-label' });
    const contextSelect = contextRow.createEl('select', { cls: 'assistant-select' });
    [4, 6, 8, 10, 12].forEach((value) => {
      const option = contextSelect.createEl('option');
      option.value = String(value);
      option.text = `${value} chunks`;
    });

    const footer = form.createDiv({ cls: 'assistant-footer' });
    footer.createDiv({ cls: 'assistant-hint', text: 'Enter to send · Shift+Enter for a new line' });
    const sendBtn = footer.createEl('button', { cls: 'assistant-send', text: 'Send', type: 'button' });


    const ensureSession = () => {
      if (!this.plugin.sessions) this.plugin.sessions = {};
      if (!this.plugin.currentSessionId || !this.plugin.sessions[this.plugin.currentSessionId]) {
        const ts = new Date();
        const id = `chat-${ts.getTime()}`;
        this.plugin.sessions[id] = {
          id,
          title: `Chat ${ts.toISOString().slice(0, 16).replace('T', ' ')}`,
          createdAt: ts.toISOString(),
          updatedAt: ts.toISOString(),
          messages: [],
          filePath: null
        };
        this.plugin.currentSessionId = id;
        this.plugin.saveSettings();
      }
      return this.plugin.sessions[this.plugin.currentSessionId];
    };

    const listSessions = () => {
      return Object.values(this.plugin.sessions || {}).sort((a, b) => {
        return (b.updatedAt || '').localeCompare(a.updatedAt || '');
      });
    };

    const renderSessionList = () => {
      const sessions = listSessions();
      sessionSelect.empty();
      sessions.forEach((s) => {
        const opt = sessionSelect.createEl('option');
        opt.value = s.id;
        opt.text = s.title || s.id;
      });
      if (this.plugin.currentSessionId) {
        sessionSelect.value = this.plugin.currentSessionId;
      }
    };

    const renderSession = (session) => {
      chat.empty();
      (session.messages || []).forEach((m) => {
        addMessage(m.role, m.text, m.sources || [], true);
      });
    };

    const ensureChatFolder = async () => {
      if (!this.plugin.settings.saveChatsToVault) return;
      let folder = this.plugin.settings.chatsFolder || 'z-Bev Chats';
      if (!this.plugin.settings.chatsFolder || this.plugin.settings.chatsFolder === 'Obsidian Assistant Chats') {
        folder = 'z-Bev Chats';
        this.plugin.settings.chatsFolder = folder;
        await this.plugin.saveSettings();
      }
      try {
        if (!(await this.app.vault.adapter.exists(folder))) {
          await this.app.vault.createFolder(folder);
        }
      } catch (e) {
        // ignore
      }
    };

    const DEFAULT_TITLE_RE = /^Chat \d{4}-\d{2}-\d{2}/;
    const AUTO_TITLE_MIN_MESSAGES = 6;

    const sanitizeTitle = (title) => {
      return (title || '')
        .replace(/[^A-Za-z0-9 _-]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
    };

    const buildSessionFilePath = (session) => {
      const folder = this.plugin.settings.chatsFolder || 'z-Bev Chats';
      const safeTitle = sanitizeTitle(session.title) || session.id;
      return `${folder}/${safeTitle}-${session.id}.md`;
    };

    const renameChatFile = async (session) => {
      if (!this.plugin.settings.saveChatsToVault) return;
      await ensureChatFolder();
      const newPath = buildSessionFilePath(session);
      if (!session.filePath) {
        session.filePath = newPath;
        return;
      }
      if (session.filePath === newPath) return;
      try {
        if (await this.app.vault.adapter.exists(session.filePath)) {
          await this.app.vault.adapter.rename(session.filePath, newPath);
        }
      } catch (e) {
        // ignore rename errors
      }
      session.filePath = newPath;
    };

    const saveChatToVault = async (session) => {
      if (!this.plugin.settings.saveChatsToVault) return;
      const folder = this.plugin.settings.chatsFolder || 'z-Bev Chats';
      try {
        if (!(await this.app.vault.adapter.exists(folder))) {
          await this.app.vault.createFolder(folder);
        }
      } catch (e) {
        // ignore
      }
      if (!session.filePath) {
        session.filePath = buildSessionFilePath(session);
      }
      const lines = [];
      lines.push('---');
      lines.push(`chat_id: ${session.id}`);
      lines.push(`created: ${session.createdAt}`);
      lines.push(`updated: ${session.updatedAt}`);
      lines.push('---');
      lines.push('');
      lines.push(`# ${session.title}`);
      lines.push('');
      for (const m of session.messages || []) {
        const who = m.role === 'user' ? 'You' : 'Assistant';
        lines.push(`## ${who} (${m.ts})`);
        lines.push(m.text || '');
        if (m.sources && m.sources.length) {
          lines.push('');
          lines.push(`Sources: ${m.sources.map(s => `${s.label} ${s.path}`).join(' | ')}`);
        }
        lines.push('');
      }
      const content = lines.join('\n');
      try {
        if (await this.app.vault.adapter.exists(session.filePath)) {
          await this.app.vault.adapter.write(session.filePath, content);
        } else {
          await this.app.vault.create(session.filePath, content);
        }
      } catch (e) {
        // ignore
      }
    };

    const recordMessage = async (role, text, sources) => {
      const session = ensureSession();
      const now = new Date().toISOString();
      session.updatedAt = now;
      session.messages = session.messages || [];
      session.messages.push({ role, text, sources: sources || [], ts: now });
      await this.plugin.saveSettings();
      await saveChatToVault(session);
    };

    const stripMarkdownForTitle = (text) => {
      let t = String(text || '');
      t = t.replace(/```[\s\S]*?```/g, ' ');
      t = t.replace(/`[^`]*`/g, ' ');
      t = t.replace(/\$[^$]*\$/g, ' ');
      t = t.replace(/!\[[^\]]*\]\([^)]+\)/g, ' ');
      t = t.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
      t = t.replace(/^#+\s+/gm, '');
      t = t.replace(/^>\s+/gm, '');
      t = t.replace(/^\s*[-*+]\s+/gm, '');
      t = t.replace(/\s+/g, ' ').trim();
      return t;
    };

    const makeLocalTitle = (session) => {
      if (!session || !session.messages) return '';
      const lastUser = [...session.messages].reverse().find((m) => m.role === 'user' && (m.text || '').trim());
      let base = lastUser ? lastUser.text : (session.messages.slice(-1)[0]?.text || '');
      base = stripMarkdownForTitle(base);
      if (!base) return '';
      const words = base.split(/\s+/).slice(0, 8);
      if (words.length < 2) return 'General chat';
      let title = words.join(' ');
      if (title.length > 60) title = title.slice(0, 60).trim();
      return title;
    };

    const maybeAutoTitle = async (session) => {
      if (!session || session.autoTitleApplied) return;
      if (!DEFAULT_TITLE_RE.test(session.title || '')) return;
      if ((session.messages || []).length < AUTO_TITLE_MIN_MESSAGES) return;
      const title = makeLocalTitle(session);
      if (!title) return;
      const dateStr = new Date().toISOString().slice(0, 10);
      session.title = `${dateStr} - ${title}`;
      session.autoTitleApplied = true;
      await renameChatFile(session);
      await this.plugin.saveSettings();
      renderSessionList();
      await saveChatToVault(session);
    };

    let assistantReplyCount = 0;
    const feedbackEvery = 4;

    const sendFeedback = async (helpful, comment) => {
      try {
        await fetch(`${this.plugin.settings.serverUrl}/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ helpful, comment })
        });
      } catch (e) {
        // ignore
      }
    };

    const addFeedbackBar = (msg, text) => {
      const row = msg.createDiv({ cls: 'assistant-feedback' });
      row.createDiv({ cls: 'assistant-feedback-label', text: 'Was this helpful?' });
      const yesBtn = row.createEl('button', { cls: 'assistant-feedback-btn', text: 'Yes', type: 'button' });
      const noBtn = row.createEl('button', { cls: 'assistant-feedback-btn', text: 'No', type: 'button' });

      const markDone = () => {
        yesBtn.disabled = true;
        noBtn.disabled = true;
        row.createDiv({ cls: 'assistant-feedback-done', text: 'Thanks!' });
      };

      yesBtn.addEventListener('click', () => {
        sendFeedback(true, text);
        markDone();
      });
      noBtn.addEventListener('click', () => {
        sendFeedback(false, text);
        markDone();
      });
    };

    const renderMarkdown = (markdown, target) => {
      target.empty();
      try {
        const result = MarkdownRenderer.renderMarkdown(markdown || '', target, '', this);
        if (result && typeof result.then === 'function') {
          result.catch(() => target.setText(markdown || ''));
        }
      } catch (err) {
        target.setText(markdown || '');
      }
    };

    const BEV_POSE_CLASSES = {
      happy: 'bev-pose-0',
      curious: 'bev-pose-1',
      working: 'bev-pose-2',
      idea: 'bev-pose-3',
      sleepy: 'bev-pose-4',
      reading: 'bev-pose-5',
      proud: 'bev-pose-6',
      coffee: 'bev-pose-7',
      confused: 'bev-pose-8'
    };

    const BEV_POSE_CLASS_LIST = Object.values(BEV_POSE_CLASSES);

    const setBevPose = (avatarEl, poseName) => {
      const cls = BEV_POSE_CLASSES[poseName] || BEV_POSE_CLASSES.coffee;
      avatarEl.classList.remove(...BEV_POSE_CLASS_LIST);
      avatarEl.classList.add(cls);
      avatarEl.dataset.pose = poseName;
    };

    const pickBevPose = (text, sources, pending) => {
      if (pending) return 'curious';
      const lower = (text || '').toLowerCase();
      if (/error|fail|failed|cannot|can't|unable|sorry|issue|problem|invalid|unknown/i.test(lower)) {
        return 'confused';
      }
      if (/sleep|tired|nap|zzz/i.test(lower)) return 'sleepy';
      if (/read|reading|note|notes|paper|book|summary|summarize|sources/i.test(lower)) return 'reading';
      if (/idea|suggest|plan|approach|proposal|recommend|next step/i.test(lower)) return 'idea';
      if (/done|fixed|success|completed|applied|ready|works/i.test(lower)) return 'proud';
      if (/code|command|run|running|terminal|script|debug|stack|log/i.test(lower) || /```/.test(lower)) {
        return 'working';
      }
      if (/coffee|break|relax|chill/i.test(lower)) return 'coffee';
      if (/thanks|thank you|great|glad|happy|love|nice/i.test(lower)) return 'happy';
      if (sources && sources.length) return 'reading';
      return 'coffee';
    };

    const createMessageRow = (role, text, sources, options = {}) => {
      const row = chat.createDiv({ cls: `assistant-row assistant-row-${role}` });
      row.setAttr('data-role', role);
      let msg = row;
      if (role === 'assistant') {
        const avatar = row.createDiv({ cls: 'assistant-avatar' });
        const pose = options.pose || pickBevPose(text, sources, options.pending);
        setBevPose(avatar, pose);
        msg = row.createDiv({ cls: 'assistant-msg assistant-assistant' });
      } else {
        msg = row.createDiv({ cls: 'assistant-msg assistant-user' });
      }
      return { row, msg };
    };

    const addMessage = (role, text, sources, fromHistory = false) => {
      const { row, msg } = createMessageRow(role, text, sources, {
        pending: text === 'Thinking...'
      });
      const label = msg.createDiv({ cls: 'assistant-role' });
      label.setText(role === 'user' ? 'You' : 'Bev');
      const textEl = msg.createDiv({ cls: 'assistant-text' });
      renderMarkdown(text, textEl);

      if (role === 'assistant') {
        const copyBtn = msg.createEl('button', { text: 'Copy', cls: 'assistant-copy' });
        copyBtn.addEventListener('click', () => navigator.clipboard.writeText(text));
      }

      if (sources && sources.length) {
        const meta = msg.createDiv({ cls: 'assistant-meta' });
        meta.setText('Sources: ' + sources.map(s => `${s.label} ${s.path}`).join(' | '));
      }

      if (!fromHistory && role === 'assistant' && text && text !== 'Thinking...') {
        assistantReplyCount += 1;
        if (assistantReplyCount % feedbackEvery === 0) {
          addFeedbackBar(msg, text);
        }
      }

      chat.scrollTop = chat.scrollHeight;
      return row;
    };

    let pendingEl = null;
    const setThinking = (on) => {
      if (on) {
        if (!pendingEl) {
          pendingEl = addMessage('assistant', 'Thinking...');
          pendingEl.classList.add('assistant-pending');
        }
        return;
      }
      if (pendingEl) {
        pendingEl.remove();
        pendingEl = null;
      }
    };

    const setComposeMode = () => {
      if (editToggle.checked) {
        input.placeholder = 'Describe the edit you want to apply...';
        container.classList.add('is-editing');
      } else {
        input.placeholder = 'Ask about your vault or papers...';
        container.classList.remove('is-editing');
      }
    };

    let serverOnline = false;
    const setServerStatus = (state, text) => {
      statusEl.setText(text);
      statusEl.dataset.state = state;
      serverOnline = state === 'online';
      sendBtn.disabled = !serverOnline;
      input.disabled = !serverOnline;
    };
    const isConnectionError = (text) => /failed to fetch|connection refused|econnrefused/i.test(text || '');

    const getActiveNotePath = () => {
      const active = this.app.workspace.getActiveFile();
      return active && active.path ? active.path : '';
    };

    const updateActiveNoteIndicator = () => {
      const active = this.app.workspace.getActiveFile();
      if (!active || !active.path) {
        activeName.setText('(none)');
        activeRow.classList.add('is-empty');
        return;
      }
      activeName.setText(active.path);
      activeRow.classList.remove('is-empty');
    };

    const addDiff = (diffText, token, path) => {
      const { msg } = createMessageRow('assistant', 'Proposed edit', null, { pose: 'working' });
      msg.createDiv({ cls: 'assistant-role', text: 'Bev' });
      msg.createDiv({ text: `Proposed edit for ${path}`, cls: 'assistant-diff-title' });
      const pre = msg.createEl('pre', { cls: 'assistant-diff' });
      pre.setText(diffText || '(no changes)');
      const actions = msg.createDiv({ cls: 'assistant-actions' });
      const applyBtn = actions.createEl('button', { text: 'Apply', cls: 'assistant-apply' });
      const rejectBtn = actions.createEl('button', { text: 'Reject', cls: 'assistant-reject' });

      applyBtn.addEventListener('click', async () => {
        const resp = await fetch(`${this.plugin.settings.serverUrl}/edit/apply`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token })
        });
        if (!resp.ok) {
          const text = await resp.text();
          addMessage('assistant', `Apply failed: ${text}`);
          return;
        }
        addMessage('assistant', 'Change applied.');
        actions.remove();
      });

      rejectBtn.addEventListener('click', async () => {
        await fetch(`${this.plugin.settings.serverUrl}/edit/reject`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token })
        });
        addMessage('assistant', 'Change rejected. No edits were applied.');
        actions.remove();
      });
    };

    const modeButtonMap = {
      safe: modeBtnSafe,
      power: modeBtnPower,
      full: modeBtnFull
    };

    const setModeUI = (mode) => {
      Object.entries(modeButtonMap).forEach(([key, btn]) => {
        btn.classList.toggle('is-active', key === mode);
      });
    };

    const setSelectValue = (selectEl, value, label) => {
      const str = String(value);
      const has = Array.from(selectEl.options).some((opt) => opt.value === str);
      if (!has) {
        const option = selectEl.createEl('option');
        option.value = str;
        option.text = `${label} (${str})`;
      }
      selectEl.value = str;
    };

    const applySettings = (data) => {
      if (data.mode) {
        setModeUI(data.mode);
      }
      readOnlyToggle.checked = !!data.read_only;
      allowEditToggle.checked = !!data.allow_edit;
      terminalToggle.checked = !!data.allow_terminal;
      pythonToggle.checked = !!data.allow_python;
      selfImproveToggle.checked = !!data.allow_self_improve;
      editToggle.disabled = !data.allow_edit;
      pathInput.disabled = !data.allow_edit;
      changeFileBtn.disabled = !data.allow_edit;
      if (!data.allow_edit) {
        editToggle.checked = false;
      }
      fileRow.classList.toggle('is-hidden', !editToggle.checked);
      if (!editToggle.checked) {
        pathWrap.classList.add('is-hidden');
      }
      setComposeMode();
      if (typeof data.max_response_tokens === 'number') {
        setSelectValue(responseSelect, data.max_response_tokens, 'Custom');
      }
      if (typeof data.max_context_chunks === 'number') {
        setSelectValue(contextSelect, data.max_context_chunks, 'Custom');
      }
    };

    const loadSettings = async () => {
      try {
        const resp = await fetch(`${this.plugin.settings.serverUrl}/settings`);
        if (!resp.ok) {
          setServerStatus('offline', 'Server offline');
          return;
        }
        const data = await resp.json();
        setServerStatus('online', 'Server online');
        applySettings(data);
      } catch (e2) {
        setServerStatus('offline', 'Server offline');
      }
    };

    const updateSettings = async (payload) => {
      try {
        const resp = await fetch(`${this.plugin.settings.serverUrl}/settings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!resp.ok) return;
        const data = await resp.json();
        setServerStatus('online', 'Server online');
        applySettings(data);
      } catch (e2) {
        setServerStatus('offline', 'Server offline');
      }
    };

    settingsToggle.addEventListener('click', () => {
      settingsPanel.classList.toggle('is-hidden');
      settingsToggle.setText(settingsPanel.classList.contains('is-hidden') ? 'Settings' : 'Hide settings');
    });

    Object.values(modeButtonMap).forEach((btn) => {
      btn.addEventListener('click', () => updateSettings({ mode: btn.dataset.mode }));
    });
    readOnlyToggle.addEventListener('change', () => updateSettings({ read_only: readOnlyToggle.checked }));
    allowEditToggle.addEventListener('change', () => updateSettings({ allow_edit: allowEditToggle.checked }));
    terminalToggle.addEventListener('change', () => updateSettings({ allow_terminal: terminalToggle.checked }));
    pythonToggle.addEventListener('change', () => updateSettings({ allow_python: pythonToggle.checked }));
    selfImproveToggle.addEventListener('change', () => updateSettings({ allow_self_improve: selfImproveToggle.checked }));
    responseSelect.addEventListener('change', () =>
      updateSettings({ max_response_tokens: parseInt(responseSelect.value, 10) })
    );
    contextSelect.addEventListener('change', () =>
      updateSettings({ max_context_chunks: parseInt(contextSelect.value, 10) })
    );

    setModeUI('safe');
    loadSettings();
    updateActiveNoteIndicator();
    this.plugin.registerEvent(this.app.workspace.on('file-open', () => updateActiveNoteIndicator()));




    const setEditUI = (enabled) => {
      fileRow.classList.toggle('is-hidden', !enabled);
      if (!enabled) {
        pathWrap.classList.add('is-hidden');
        pathInput.value = '';
        fileName.setText('(select a note)');
      }
      setComposeMode();
    };

    editToggle.addEventListener('change', () => {
      setEditUI(editToggle.checked);
      if (editToggle.checked) {
        autoFillPath('');
      }
    });

    changeFileBtn.addEventListener('click', () => {
      pathWrap.classList.toggle('is-hidden');
      if (!pathWrap.classList.contains('is-hidden')) {
        pathInput.focus();
        pathInput.select();
      }
    });

    pathInput.addEventListener('input', () => {
      const value = pathInput.value.trim();
      fileName.setText(value || '(select a note)');
    });


    newSessionBtn.addEventListener('click', async () => {
      const ts = new Date();
      const id = `chat-${ts.getTime()}`;
      this.plugin.sessions[id] = {
        id,
        title: `Chat ${ts.toISOString().slice(0, 16).replace('T', ' ')}`,
        createdAt: ts.toISOString(),
        updatedAt: ts.toISOString(),
        messages: [],
        filePath: null
      };
      this.plugin.currentSessionId = id;
      await this.plugin.saveSettings();
      renderSessionList();
      renderSession(this.plugin.sessions[id]);
    });

    sessionSelect.addEventListener('change', async () => {
      const id = sessionSelect.value;
      if (!id) return;
      this.plugin.currentSessionId = id;
      await this.plugin.saveSettings();
      const session = this.plugin.sessions[id];
      if (session) {
        renderSession(session);
      }
    });

    renderSessionList();
    renderSession(ensureSession());
    setEditUI(false);
    ensureChatFolder();


    const autoFillPath = (message) => {
      let path = pathInput.value.trim();
      if (!path) {
        path = getActiveNotePath();
      }
      if (!path && message) {
        const match = message.match(/([^\s]+\.(?:md|markdown|txt))/i);
        if (match) {
          path = match[1];
        }
      }
      if (path) {
        pathInput.value = path;
        fileName.setText(path);
      }
    };

    input.addEventListener('keydown', (evt) => {
      if (evt.key === 'Enter' && !evt.shiftKey) {
        evt.preventDefault();
        if (typeof form.requestSubmit === 'function') {
          form.requestSubmit();
        } else {
          form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        }
      }
    });

    sendBtn.addEventListener('click', () => {
      if (typeof form.requestSubmit === 'function') {
        form.requestSubmit();
      } else {
        form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      }
    });

form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;

      if (editToggle.checked) {
        if (editToggle.disabled) {
          addMessage('assistant', 'Edit is disabled in settings.');
          return;
        }
        autoFillPath(message);
        const path = pathInput.value.trim();
        if (!path) {
          addMessage('assistant', 'Please provide a note path for edit mode.');
          return;
        }
        addMessage('user', `Edit request: ${path}`);
        await recordMessage('user', `Edit request: ${path}`, []);
        input.value = '';
        try {
          const resp = await fetch(`${this.plugin.settings.serverUrl}/edit/propose`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, instruction: message })
          });
          if (!resp.ok) {
            const text = await resp.text();
            addMessage('assistant', `Edit proposal failed: ${text}`);
            return;
          }
          const data = await resp.json();
          addDiff(data.diff || '', data.token, data.path);
        } catch (e2) {
          addMessage('assistant', `Connection error: ${e2}`);
        }
        return;
      }

      addMessage('user', message);
      await recordMessage('user', message, []);
      input.value = '';
      const session = ensureSession();
      const history = (session.messages || [])
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .slice(-this.plugin.settings.maxHistoryMessages * 2)
        .map((m) => ({ role: m.role, content: m.text }));
      try {
        setThinking(true);
        const activeNotePath = getActiveNotePath();
        const resp = await fetch(`${this.plugin.settings.serverUrl}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message,
            use_web: webToggle.checked,
            history,
            active_note: activeNotePath
          })
        });
        if (!resp.ok) {
          const text = await resp.text();
          setThinking(false);
          if (isConnectionError(text)) {
            setServerStatus('offline', 'Server offline');
          }
          addMessage('assistant', `Server error: ${text}`);
          return;
        }
        const data = await resp.json();
        setThinking(false);
        setServerStatus('online', 'Server online');
        addMessage('assistant', data.reply || '', data.sources || []);
        await recordMessage('assistant', data.reply || '', data.sources || []);
        await maybeAutoTitle(session);
      } catch (e2) {
        setThinking(false);
        setServerStatus('offline', 'Server offline');
        addMessage('assistant', `Connection error: ${e2}`);
      }
    });
  }
}

class AssistantSettingTab {
  constructor(app, plugin) {
    this.app = app;
    this.plugin = plugin;
  }

  display() {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl('h2', { text: 'Obsidian Assistant' });

    new Setting(containerEl)
      .setName('Server URL')
      .setDesc('Where the local assistant server is running.')
      .addText((text) =>
        text
          .setPlaceholder('http://127.0.0.1:8000')
          .setValue(this.plugin.settings.serverUrl)
          .onChange(async (value) => {
            this.plugin.settings.serverUrl = value.trim();
            await this.plugin.saveSettings();
            this.plugin.refreshViews();
          })
      );

    new Setting(containerEl)
      .setName('Open on startup')
      .setDesc('Open the assistant pane when the vault loads.')
      .addToggle((toggle) =>
        toggle
          .setValue(this.plugin.settings.autoOpen)
          .onChange(async (value) => {
            this.plugin.settings.autoOpen = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName('Save chats to vault')
      .setDesc('Write chat transcripts to a folder in your vault.')
      .addToggle((toggle) =>
        toggle
          .setValue(this.plugin.settings.saveChatsToVault)
          .onChange(async (value) => {
            this.plugin.settings.saveChatsToVault = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName('Chats folder')
      .setDesc('Folder inside the vault to store chat transcripts.')
      .addText((text) =>
        text
          .setPlaceholder('z-Bev Chats')
          .setValue(this.plugin.settings.chatsFolder)
          .onChange(async (value) => {
            this.plugin.settings.chatsFolder = value.trim() || 'z-Bev Chats';
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName('Max history messages')
      .setDesc('How many recent messages to include for context.')
      .addText((text) =>
        text
          .setPlaceholder('8')
          .setValue(String(this.plugin.settings.maxHistoryMessages))
          .onChange(async (value) => {
            const v = parseInt(value, 10);
            this.plugin.settings.maxHistoryMessages = Number.isFinite(v) && v > 0 ? v : 8;
            await this.plugin.saveSettings();
          })
      );
  }
}

module.exports = class ObsidianAssistantPlugin extends Plugin {
  async onload() {
    await this.loadSettings();

    this.registerView(VIEW_TYPE, (leaf) => new AssistantView(leaf, this));

    this.addCommand({
      id: 'open-obsidian-assistant',
      name: 'Open Assistant Pane',
      callback: () => this.activateView()
    });

    this.addSettingTab(new AssistantSettingTab(this.app, this));

    this.app.workspace.onLayoutReady(() => {
      if (this.settings.autoOpen) {
        this.activateView();
      }
    });
  }

  onunload() {
    this.app.workspace.getLeavesOfType(VIEW_TYPE).forEach((leaf) => leaf.detach());
  }

  async activateView() {
    const existing = this.app.workspace.getLeavesOfType(VIEW_TYPE);
    if (existing.length > 0) {
      this.app.workspace.revealLeaf(existing[0]);
      return;
    }
    const leaf = this.app.workspace.getRightLeaf(false);
    await leaf.setViewState({ type: VIEW_TYPE, active: true });
    this.app.workspace.revealLeaf(leaf);
  }

  async loadSettings() {
    const data = (await this.loadData()) || {};
    if (data && data.settings) {
      this.settings = Object.assign({}, DEFAULT_SETTINGS, data.settings);
      this.sessions = data.sessions || {};
      this.currentSessionId = data.currentSessionId || null;
    } else {
      this.settings = Object.assign({}, DEFAULT_SETTINGS, data);
      this.sessions = {};
      this.currentSessionId = null;
    }

    let changed = false;
    if (this.settings.chatsFolder === 'Obsidian Assistant Chats') {
      this.settings.chatsFolder = 'z-Bev Chats';
      changed = true;
    }
    const oldPrefix = 'Obsidian Assistant Chats/';
    const newPrefix = `${this.settings.chatsFolder}/`;
    Object.values(this.sessions || {}).forEach((session) => {
      if (session && session.filePath && session.filePath.startsWith(oldPrefix)) {
        session.filePath = session.filePath.replace(oldPrefix, newPrefix);
        changed = true;
      }
    });
    if (changed) {
      await this.saveSettings();
    }
  }

  async saveSettings() {
    await this.saveData({
      settings: this.settings,
      sessions: this.sessions || {},
      currentSessionId: this.currentSessionId || null
    });
  }

  refreshViews() {
    const leaves = this.app.workspace.getLeavesOfType(VIEW_TYPE);
    for (const leaf of leaves) {
      const view = leaf.view;
      if (view && typeof view.render === 'function') {
        view.render();
      }
    }
  }
};
