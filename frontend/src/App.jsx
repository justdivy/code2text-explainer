import { useRef, useState } from 'react';
import Editor from '@monaco-editor/react';
import axios from 'axios';
import './App.css';

function FlowSteps({ steps }) {
  return (
    <div className="flow-steps">
      {steps.map((step, index) => (
        <div className="flow-item" key={`${step.type}-${index}-${step.label}`}>
          <div className={`flow-node flow-${step.type}`}>
            <span className="flow-node-icon" aria-hidden="true">
              {step.type === 'start' ? '▶' : step.type === 'end' ? '✓' : step.type === 'decision' ? '?' : step.type === 'loop' ? '↻' : step.type === 'function' ? 'ƒ' : step.type === 'class' ? 'C' : step.type === 'return' ? '↩' : '›'}
            </span>
            <span>{step.label}</span>
          </div>

          {step.type === 'decision' && (
            <div className="flow-branches">
              {step.branches.map((branch) => (
                <div className="flow-branch" key={branch.label}>
                  <span className="flow-branch-label">{branch.label}</span>
                  <FlowSteps steps={branch.steps} />
                </div>
              ))}
            </div>
          )}

          {(step.type === 'loop' || step.type === 'function' || step.type === 'class') && (
            <div className="flow-nested">
              <span className="flow-nested-label">
                {step.type === 'loop'
                  ? 'Repeat for each item'
                  : step.type === 'function'
                    ? 'Function body (when called)'
                    : 'Class contents'}
              </span>
              <FlowSteps steps={step.steps} />
            </div>
          )}

          {index < steps.length - 1 && <span className="flow-arrow" aria-hidden="true">↓</span>}
        </div>
      ))}
    </div>
  );
}

function App() {
  const [code, setCode] = useState('# Type your Python code here\ndef add(a, b):\n    return a + b');
  const [explanation, setExplanation] = useState('');
  const [flowchart, setFlowchart] = useState(null);
  const [loading, setLoading] = useState(false);
  const requestId = useRef(0);

  const handleCodeChange = (value) => {
    requestId.current += 1;
    setCode(value ?? '');
    setExplanation('');
    setFlowchart(null);
    setLoading(false);
  };

  const handleTranslate = async () => {
    if (!code.trim()) {
      setExplanation('Enter some code before requesting an explanation.');
      return;
    }

    const currentRequest = ++requestId.current;
    const submittedCode = code;
    setLoading(true);
    setExplanation('');
    setFlowchart(null);

    try {
      const response = await axios.post('http://localhost:8000/explain', {
        code: submittedCode,
      });
      if (currentRequest === requestId.current) {
        setExplanation(response.data.explanation);
        setFlowchart(response.data.flowchart);
      }
    } catch (error) {
      if (currentRequest === requestId.current) {
        setExplanation(
          error.response?.data?.detail ||
            'Could not reach the backend. Make sure the API and model are running, then try again.'
        );
      }
    } finally {
      if (currentRequest === requestId.current) {
        setLoading(false);
      }
    }
  };

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">&lt;/&gt;</span>
          <div>
            <p className="brand-eyebrow">NLP CODE STUDIO</p>
            <h1>CodeSage</h1>
          </div>
        </div>
        <button onClick={handleTranslate} disabled={loading}>
          <span aria-hidden="true">{loading ? '◌' : '✦'}</span>
          {loading ? 'Translating...' : 'Explain code'}
        </button>
      </header>

      <section className="workspace">
        <div className="editor-pane" aria-label="Python code editor">
          <div className="editor-panel">
            <div className="editor-toolbar">
              <div className="editor-heading">
                <span className="window-dots" aria-hidden="true"><i /><i /><i /></span>
                <span>CodeLens</span>
              </div>
              <span className="language-pill"><span />Python</span>
            </div>
            <div className="editor-canvas">
              <Editor
                height="100%"
                defaultLanguage="python"
                theme="vs-dark"
                value={code}
                onChange={handleCodeChange}
                options={{
                  minimap: { enabled: false },
                  fontSize: 15,
                  lineHeight: 24,
                  fontFamily: "'Cascadia Code', 'Fira Code', Consolas, monospace",
                  fontLigatures: true,
                  padding: { top: 18, bottom: 18 },
                  cursorBlinking: 'smooth',
                  cursorSmoothCaretAnimation: 'on',
                  smoothScrolling: true,
                  renderLineHighlight: 'all',
                  guides: { bracketPairs: true, indentation: true },
                  scrollBeyondLastLine: false,
                  roundedSelection: true,
                }}
              />
            </div>
            <div className="editor-statusbar">
              <span><i />Ready to edit</span>
              <span>Python</span>
            </div>
          </div>
        </div>

        <div className="result-pane">
          <div className="result-heading">
            <span className="result-icon" aria-hidden="true">✧</span>
            <div>
              <p>YOUR RESULTS</p>
              <h2>English explanation</h2>
            </div>
          </div>
          <div className="explanation-card" aria-live="polite">
            {explanation ? (
              <p>{explanation}</p>
            ) : (
              <p className="empty-state">
                {loading
                  ? 'Reading the current code...'
                  : 'Edit the code, then click Translate Code to explain the current version.'}
              </p>
            )}
          </div>

          <section className="flowchart-section" aria-label="Code execution flowchart">
            <div className="flowchart-heading">
              <span className="flowchart-heading-icon" aria-hidden="true">⌘</span>
              <div>
                <p>STEP BY STEP</p>
                <h3>Execution flow</h3>
              </div>
            </div>
            <div className="flowchart-card">
              {flowchart?.error ? (
                <p className="flowchart-message">{flowchart.error}</p>
              ) : flowchart?.steps?.length ? (
                <FlowSteps steps={flowchart.steps} />
              ) : (
                <p className="flowchart-message">
                  Translate the current code to see its execution flow.
                </p>
              )}
            </div>
            <p className="flowchart-note">This diagram describes the code; it does not run it.</p>
          </section>
        </div>
      </section>
    </main>
  );
}

export default App;
