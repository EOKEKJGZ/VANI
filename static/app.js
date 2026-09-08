/**
 * app.js — Frontend Controller for Sahakaar Sathi VANI Platform
 * SIH Problem Statement 26088 — Ministry of Cooperation / NCCT
 * Web Speech API (STT & TTS), Rich Markdown-to-Ledger Renderer, REST API Client
 */

(function () {
  'use strict';

  // DOM Elements
  const chatStream = document.getElementById('chat-stream');
  const queryForm = document.getElementById('query-form');
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const voiceMicBtn = document.getElementById('voice-mic-btn');
  const micIcon = document.getElementById('mic-icon');
  const liveDot = document.getElementById('live-dot');
  const voiceStatusText = document.getElementById('voice-status-text');
  const activeLangIndicator = document.getElementById('active-lang-indicator');
  const clearHistoryBtn = document.getElementById('clear-history-btn');
  const emergencyCropLossBtn = document.getElementById('emergency-crop-loss-btn');
  const openSchemesBtn = document.getElementById('open-schemes-btn');
  const schemesModal = document.getElementById('schemes-modal');
  const modalCloseBtn = document.getElementById('modal-close-btn');
  const schemeSearchBox = document.getElementById('scheme-search-box');
  const schemesResultsGrid = document.getElementById('schemes-results-grid');
  const langTabs = document.querySelectorAll('.lang-tab');
  const fontDecBtn = document.getElementById('font-dec-btn');
  const fontDefBtn = document.getElementById('font-def-btn');
  const fontIncBtn = document.getElementById('font-inc-btn');
  const categoryTiles = document.querySelectorAll('.tile-card');

  // State
  let currentLang = 'en-IN';
  let currentLangLabel = 'English';
  let isListening = false;
  let recognition = null;
  let currentUtterance = null;
  let currentSpeakingBtn = null;
  let isSubmitting = false;

  // ------------------------------------------------------------------
  // 1. Accessibility & Toolbar Controls
  // ------------------------------------------------------------------

  // Language Selection Tabs
  langTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      langTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentLang = tab.dataset.lang || 'en-IN';
      currentLangLabel = tab.dataset.label || 'English';
      activeLangIndicator.textContent = `Active Language: ${currentLangLabel} (${currentLang})`;
      if (recognition) {
        recognition.lang = currentLang;
      }
      voiceStatusText.textContent = `Language switched to ${currentLangLabel}. Microphone ready.`;
    });
  });

  // Font Size Scaler (A- / A / A+)
  fontDecBtn.addEventListener('click', () => {
    document.documentElement.setAttribute('data-font-size', 'sm');
    fontDecBtn.classList.add('active');
    fontDefBtn.classList.remove('active');
    fontIncBtn.classList.remove('active');
  });

  fontDefBtn.addEventListener('click', () => {
    document.documentElement.setAttribute('data-font-size', 'md');
    fontDecBtn.classList.remove('active');
    fontDefBtn.classList.add('active');
    fontIncBtn.classList.remove('active');
  });

  fontIncBtn.addEventListener('click', () => {
    document.documentElement.setAttribute('data-font-size', 'lg');
    fontDecBtn.classList.remove('active');
    fontDefBtn.classList.remove('active');
    fontIncBtn.classList.add('active');
  });

  // ------------------------------------------------------------------
  // 2. Web Speech API (Speech-to-Text)
  // ------------------------------------------------------------------

  function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      voiceStatusText.textContent = 'Speech recognition not supported in this browser. Please type your query.';
      voiceMicBtn.disabled = false;
      return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = currentLang;

    recognition.onstart = () => {
      isListening = true;
      voiceMicBtn.classList.add('listening');
      liveDot.classList.add('active');
      voiceStatusText.textContent = `Listening in ${currentLangLabel}... Please speak now.`;
    };

    recognition.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      if (interimTranscript) {
        userInput.value = interimTranscript;
        voiceStatusText.textContent = `Hearing: "${interimTranscript}"...`;
      }

      if (finalTranscript) {
        userInput.value = finalTranscript.trim();
        voiceStatusText.textContent = `Recognized: "${finalTranscript.trim()}". Submitting...`;
        stopListening();
        // Automatically submit query on speech completion
        setTimeout(() => {
          submitQuery(finalTranscript.trim());
        }, 300);
      }
    };

    recognition.onerror = (event) => {
      console.warn('[Speech Error]', event.error);
      stopListening();
      if (event.error === 'not-allowed') {
        voiceStatusText.textContent = 'Microphone permission denied. Please allow microphone access.';
      } else {
        voiceStatusText.textContent = `Voice recognition error (${event.error}). Please try again or type.`;
      }
    };

    recognition.onend = () => {
      stopListening();
    };
  }

  function startListening() {
    if (!recognition) {
      initSpeechRecognition();
    }
    if (recognition) {
      try {
        recognition.lang = currentLang;
        recognition.start();
      } catch (err) {
        console.warn('Recognition start issue:', err);
      }
    }
  }

  function stopListening() {
    isListening = false;
    voiceMicBtn.classList.remove('listening');
    liveDot.classList.remove('active');
    if (recognition) {
      try {
        recognition.stop();
      } catch (e) {}
    }
    if (!voiceStatusText.textContent.includes('Submitting')) {
      voiceStatusText.textContent = `Microphone ready &bull; Tap button to speak in ${currentLangLabel}`;
    }
  }

  voiceMicBtn.addEventListener('click', () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  });

  // ------------------------------------------------------------------
  // 3. Text-to-Speech (Audio Readout Synthesis)
  // ------------------------------------------------------------------

  window.speakLedgerCard = function (button) {
    if (!('speechSynthesis' in window)) {
      alert('Audio readout is not supported on this browser.');
      return;
    }

    // If currently playing, stop it
    if (window.speechSynthesis.speaking && currentSpeakingBtn === button) {
      window.speechSynthesis.cancel();
      resetAudioButtons();
      return;
    }

    window.speechSynthesis.cancel();
    resetAudioButtons();

    // Extract text content from the message card
    const card = button.closest('.ledger-row').querySelector('.row-card');
    if (!card) return;

    // Clean text: remove raw table formatting or symbols for smooth audio
    let cleanText = card.innerText
      .replace(/\|/g, ' ')
      .replace(/---/g, '')
      .replace(/###/g, '')
      .replace(/##/g, '')
      .replace(/\*/g, '')
      .trim();

    if (!cleanText) return;

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = currentLang;
    utterance.rate = 0.95; // Slightly slower, clear rate for elderly & rural users
    utterance.pitch = 1.0;

    button.classList.add('playing');
    button.innerHTML = '<i class="ti ti-player-stop"></i> <span>Stop</span>';
    currentSpeakingBtn = button;

    utterance.onend = () => {
      resetAudioButtons();
    };

    utterance.onerror = () => {
      resetAudioButtons();
    };

    window.speechSynthesis.speak(utterance);
  };

  function resetAudioButtons() {
    document.querySelectorAll('.row-audio-control').forEach(btn => {
      btn.classList.remove('playing');
      btn.innerHTML = '<i class="ti ti-volume"></i> <span>Listen</span>';
    });
    currentSpeakingBtn = null;
  }

  // ------------------------------------------------------------------
  // 4. Rich Markdown-to-Ledger Formatter
  // ------------------------------------------------------------------

  function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return text.replace(/[&<>"']/g, m => map[m]);
  }

  function formatMarkdownResponse(markdown) {
    if (!markdown) return '';

    let html = markdown;

    // 1. Process Markdown Tables
    const tableRegex = /((?:\|[^\n]+\|\r?\n)+)/g;
    html = html.replace(tableRegex, (match) => {
      const rows = match.trim().split(/\r?\n/).map(r => r.trim()).filter(r => r.startsWith('|') && r.endsWith('|'));
      if (rows.length < 2) return match;

      let tableHtml = '<div class="table-wrapper"><table class="parsed-table">';
      let isHeader = true;

      for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        // Check if separator line
        if (/^\|(?:\s*:?-+:?\s*\|)+$/.test(row)) {
          isHeader = false;
          continue;
        }

        const cells = row.split('|').slice(1, -1).map(c => c.trim());
        tableHtml += '<tr>';
        cells.forEach(cell => {
          const tag = isHeader ? 'th' : 'td';
          tableHtml += `<${tag}>${formatInlineMarkdown(cell)}</${tag}>`;
        });
        tableHtml += '</tr>';
      }
      tableHtml += '</table></div>';
      return tableHtml;
    });

    // 2. Emergency 72-Hour Warning or Urgent Callouts
    html = html.replace(/(?:🚨|\*\*🚨|\*\*Warning|\*\*Emergency)(.*?)(?:\n|$)/gi, (match, p1) => {
      return `<div class="response-callout urgent"><i class="ti ti-alert-triangle" style="font-size:1.3rem;"></i><div><strong>CRITICAL DEADLINE / ACTION:</strong> ${formatInlineMarkdown(p1)}</div></div>`;
    });

    // 3. Action Step Cards (e.g. "1. **Report loss** ...")
    html = html.replace(/^(\d+)\.\s*\*\*(.*?)\*\*\s*(.*?)$/gm, (match, num, title, desc) => {
      return `<div class="action-step-item"><span class="step-num-badge">${num}</span><div><strong style="color:var(--indigo-dark);">${formatInlineMarkdown(title)}</strong>: ${formatInlineMarkdown(desc)}</div></div>`;
    });

    // 4. Section Headings (### Title)
    html = html.replace(/^###\s+(.*?)$/gm, '<h3>$1</h3>');
    html = html.replace(/^##\s+(.*?)$/gm, '<h2>$1</h2>');
    html = html.replace(/^#\s+(.*?)$/gm, '<h2>$1</h2>');

    // 5. Bullet Lists (- Item or • Item)
    html = html.replace(/^[•\-\*]\s+(.*?)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*?<\/li>\s*)+)/g, '<ul>$1</ul>');

    // 6. Horizontal Rules
    html = html.replace(/^---$/gm, '<hr style="border:0; border-top:1px solid var(--border-color); margin:14px 0;">');

    // 7. Inline Formatting (bold, italic, code, links)
    html = formatInlineMarkdown(html);

    // 8. Paragraphs
    const blocks = html.split(/\n{2,}/);
    html = blocks.map(block => {
      const trimmed = block.trim();
      if (!trimmed) return '';
      if (trimmed.startsWith('<div') || trimmed.startsWith('<table') || trimmed.startsWith('<ul') || trimmed.startsWith('<ol') || trimmed.startsWith('<h2') || trimmed.startsWith('<h3') || trimmed.startsWith('<hr')) {
        return trimmed;
      }
      return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
    }).join('\n');

    return html;
  }

  function formatInlineMarkdown(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code style="background:var(--surface-inset); padding:2px 6px; border-radius:4px; font-size:0.9em;">$1</code>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" style="color:var(--terracotta); font-weight:600; text-decoration:underline;">$1</a>');
  }

  // ------------------------------------------------------------------
  // 5. Chat UI Streaming & Interaction
  // ------------------------------------------------------------------

  function appendUserMessage(text) {
    const row = document.createElement('article');
    row.className = 'ledger-row user-row';
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    row.innerHTML = `
      <div class="row-card">
        ${escapeHtml(text)}
      </div>
      <div class="row-meta">
        <i class="ti ti-user"></i>
        <span>Citizen Query &bull; ${timeStr}</span>
      </div>
    `;

    chatStream.appendChild(row);
    scrollToBottom();
  }

  function appendAssistantPlaceholder() {
    const row = document.createElement('article');
    row.className = 'ledger-row assistant-row loading-row';
    row.id = 'assistant-loading-row';

    row.innerHTML = `
      <div class="row-card parsed-response" style="display:flex; align-items:center; gap:12px; padding:18px;">
        <span class="spinner" style="border-top-color:var(--indigo); width:24px; height:24px;"></span>
        <span style="color:var(--indigo-dark); font-weight:600;">Sahakaar Sathi is retrieving verified cooperative laws & government schemes database...</span>
      </div>
    `;

    chatStream.appendChild(row);
    scrollToBottom();
  }

  function removeAssistantPlaceholder() {
    const loadingRow = document.getElementById('assistant-loading-row');
    if (loadingRow) {
      loadingRow.remove();
    }
  }

  function appendAssistantMessage(rawAnswer) {
    removeAssistantPlaceholder();

    const row = document.createElement('article');
    row.className = 'ledger-row assistant-row';
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const formattedHtml = formatMarkdownResponse(rawAnswer);

    row.innerHTML = `
      <div class="row-card parsed-response">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; border-bottom:1px solid var(--brass-light); padding-bottom:6px;">
          <strong style="color:var(--indigo-dark); font-family:var(--font-serif); font-size:var(--font-size-base); display:flex; align-items:center; gap:6px;">
            <i class="ti ti-building-bank"></i> Sahakaar Sathi Advisory
          </strong>
          <span class="verified-source-badge">
            <i class="ti ti-shield-check"></i> Ministry of Cooperation &bull; NCCT Verified
          </span>
        </div>
        ${formattedHtml}
      </div>
      <div class="row-meta">
        <span>Official Advisory Response &bull; ${timeStr}</span>
        <button class="row-audio-control" onclick="window.speakLedgerCard(this)" title="Listen to response">
          <i class="ti ti-volume"></i> <span>Listen</span>
        </button>
      </div>
    `;

    chatStream.appendChild(row);
    scrollToBottom();
  }

  function scrollToBottom() {
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  async function submitQuery(queryText) {
    const text = (queryText || userInput.value || '').trim();
    if (!text || isSubmitting) return;

    isSubmitting = true;
    sendBtn.disabled = true;
    userInput.value = '';

    appendUserMessage(text);
    appendAssistantPlaceholder();

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text, chat_id: 1 })
      });

      if (!resp.ok) {
        throw new Error(`Server returned error ${resp.status}`);
      }

      const data = await resp.json();
      appendAssistantMessage(data.answer);
    } catch (err) {
      console.error('[Chat API Error]', err);
      appendAssistantMessage(`⚠️ **System Notice**: Unable to connect to the Tahsil Hub RAG server. Error: ${err.message}. Please check that server.py is running.`);
    } finally {
      isSubmitting = false;
      sendBtn.disabled = false;
      userInput.focus();
    }
  }

  // Form Submit Event
  queryForm.addEventListener('submit', (e) => {
    e.preventDefault();
    submitQuery();
  });

  // Emergency Crop Loss Button
  emergencyCropLossBtn.addEventListener('click', () => {
    submitQuery('There was a heavy rainfall yesterday all my crops are destroyed what should I do');
  });

  // Quick Action Tiles
  categoryTiles.forEach(tile => {
    tile.addEventListener('click', () => {
      const prompt = tile.dataset.prompt;
      if (prompt) {
        submitQuery(prompt);
      }
    });
  });

  // ------------------------------------------------------------------
  // 6. Session Management & History Loading
  // ------------------------------------------------------------------

  async function loadHistory() {
    try {
      const res = await fetch('/api/history?limit=25');
      if (!res.ok) return;
      const data = await res.json();
      if (data.history && data.history.length > 0) {
        chatStream.innerHTML = '';
        data.history.forEach(msg => {
          if (msg.role === 'user') {
            appendUserMessage(msg.content);
          } else if (msg.role === 'assistant') {
            appendAssistantMessage(msg.content);
          }
        });
      }
    } catch (e) {
      console.warn('History load issue:', e);
    }
  }

  clearHistoryBtn.addEventListener('click', async () => {
    if (confirm('Do you want to clear conversation history and start a new session?')) {
      try {
        await fetch('/api/clear', { method: 'POST' });
        window.speechSynthesis.cancel();
        resetAudioButtons();
        chatStream.innerHTML = '';
        // Re-inject initial welcome message
        const welcomeHtml = `
          <article class="ledger-row assistant-row">
            <div class="row-card parsed-response">
              <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; border-bottom:1px solid var(--brass-light); padding-bottom:6px;">
                <strong style="color:var(--indigo-dark); font-family:var(--font-serif); font-size:var(--font-size-base); display:flex; align-items:center; gap:6px;">
                  <i class="ti ti-building-bank"></i> Sahakaar Sathi VANI
                </strong>
                <span class="verified-source-badge">
                  <i class="ti ti-shield-check"></i> Ministry of Cooperation &bull; NCCT Verified
                </span>
              </div>
              <p><strong>New Session Initialized.</strong> I am ready to assist with PMFBY crop loss claims, PACS by-laws, Kisan Credit Card loans, and 3,400+ government welfare schemes.</p>
            </div>
            <div class="row-meta">
              <span>Official Advisory Channel &bull; Tahsil Hub</span>
              <button class="row-audio-control" onclick="window.speakLedgerCard(this)" title="Read aloud">
                <i class="ti ti-volume"></i> <span>Listen</span>
              </button>
            </div>
          </article>
        `;
        chatStream.innerHTML = welcomeHtml;
      } catch (err) {
        alert('Could not clear history: ' + err.message);
      }
    }
  });

  // ------------------------------------------------------------------
  // 7. Scheme Explorer Modal & Search
  // ------------------------------------------------------------------

  openSchemesBtn.addEventListener('click', () => {
    schemesModal.classList.add('open');
    loadSchemesList('');
  });

  modalCloseBtn.addEventListener('click', () => {
    schemesModal.classList.remove('open');
  });

  schemesModal.addEventListener('click', (e) => {
    if (e.target === schemesModal) {
      schemesModal.classList.remove('open');
    }
  });

  let searchTimeout = null;
  schemeSearchBox.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      loadSchemesList(e.target.value.trim());
    }, 250);
  });

  async function loadSchemesList(query) {
    schemesResultsGrid.innerHTML = '<div style="padding:20px; color:var(--text-muted);"><span class="spinner" style="border-top-color:var(--indigo);"></span> Loading verified schemes database...</div>';
    try {
      const resp = await fetch(`/api/schemes/search?q=${encodeURIComponent(query)}&limit=15`);
      const data = await resp.json();
      if (!data.results || data.results.length === 0) {
        schemesResultsGrid.innerHTML = '<div style="padding:20px; color:var(--text-muted);">No schemes found matching your search. Try keywords like PMFBY, fertilizer, loan, tractor, women, kisan.</div>';
        return;
      }

      schemesResultsGrid.innerHTML = '';
      data.results.forEach(scheme => {
        const card = document.createElement('div');
        card.className = 'scheme-record-card';
        card.innerHTML = `
          <div class="scheme-record-title">${escapeHtml(scheme.scheme_name)}</div>
          <div class="scheme-record-meta">
            <span style="background:var(--indigo-subtle); color:var(--indigo); padding:2px 6px; border-radius:3px; font-weight:600;">${escapeHtml(scheme.level)}</span>
            <span style="background:var(--brass-subtle); color:#59441D; padding:2px 6px; border-radius:3px;">${escapeHtml(scheme.category)}</span>
          </div>
          <p class="scheme-record-snippet">${escapeHtml(scheme.text)}</p>
          <button class="scheme-ask-btn" onclick="window.askScheme('${escapeHtml(scheme.scheme_name.replace(/'/g, "\\'"))}')">
            <i class="ti ti-messages"></i> Ask About This Scheme
          </button>
        `;
        schemesResultsGrid.appendChild(card);
      });
    } catch (err) {
      schemesResultsGrid.innerHTML = `<div style="padding:20px; color:var(--alert-rust);">Error loading schemes: ${err.message}</div>`;
    }
  }

  window.askScheme = function (schemeName) {
    schemesModal.classList.remove('open');
    submitQuery(`Tell me all eligibility criteria, benefits, and how to apply for ${schemeName}`);
  };

  // ------------------------------------------------------------------
  // 8. Initialization
  // ------------------------------------------------------------------

  initSpeechRecognition();
  loadHistory();

})();
