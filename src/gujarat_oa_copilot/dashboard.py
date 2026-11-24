"""
Generates a self-contained interactive HTML Dashboard (Chart.js) from simulation results.
"""

import json
import os
from typing import Dict, Any


def generate_html_dashboard(results: Dict[str, Any], output_html_path: str) -> str:
    """Writes an interactive Chart.js HTML report for the Gujarat OA AI Co-Pilot demo."""
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    data_json = json.dumps(results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Gujarat Open Access + IEX AI Bidding Co-Pilot</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --emerald: #10b981;
      --blue: #3b82f6;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --violet: #8b5cf6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px 32px;
      line-height: 1.5;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }}
    h1 {{ font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; }}
    .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-top: 4px; }}
    .controls {{ display: flex; gap: 12px; align-items: center; }}
    select {{
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
    }}
    .kpi-label {{ color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .kpi-value {{ font-size: 1.65rem; font-weight: 700; margin-top: 6px; }}
    .kpi-sub {{ font-size: 0.82rem; color: var(--muted); margin-top: 4px; }}
    .emerald {{ color: var(--emerald); }}
    .blue {{ color: var(--blue); }}
    .amber {{ color: var(--amber); }}
    .charts-grid {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    @media (max-width: 1024px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
    .chart-title {{ font-size: 1rem; font-weight: 600; margin-bottom: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
    th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid var(--border); }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: rgba(15,23,42,0.4); }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-oa {{ background: rgba(16, 185, 129, 0.15); color: #34d399; }}
    .badge-discom {{ background: rgba(244, 63, 94, 0.15); color: #fb7185; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>⚡ Gujarat Open Access (GERC) + IEX AI Bidding Co-Pilot</h1>
      <div class="subtitle">Backtested across 11,616 15-min blocks (Jun–Sep 2025) • Trained on 83,966 blocks (2023–2025) from IEXInsiderMCP</div>
    </div>
    <div class="controls">
      <label for="scenarioSelect" style="font-size:0.85rem;color:var(--muted);">Regulatory Profile:</label>
      <select id="scenarioSelect" onchange="renderScenario(this.value)">
        <option value="66kV_Standard_OA">66 kV EHV Standard OA (GETCO Connected)</option>
        <option value="11_33kV_Standard_OA">11/33 kV HT Standard OA (DISCOM Network)</option>
        <option value="11_33kV_Green_Captive_Blend">11/33 kV Green / Captive OA Waiver</option>
      </select>
    </div>
  </header>

  <div class="kpi-grid">
    <div class="card">
      <div class="kpi-label">DISCOM HTP-I Baseline</div>
      <div class="kpi-value" id="kpiDiscomRate">₹7.47 / kWh</div>
      <div class="kpi-sub" id="kpiDiscomBill">Monthly Bill: ₹268.95 Lakhs</div>
    </div>
    <div class="card">
      <div class="kpi-label">AI Co-Pilot + Load Shift Rate</div>
      <div class="kpi-value emerald" id="kpiAiRate">₹6.53 / kWh</div>
      <div class="kpi-sub" id="kpiAiBill">Monthly Bill: ₹234.96 Lakhs</div>
    </div>
    <div class="card">
      <div class="kpi-label">Net Monthly & Annual Savings</div>
      <div class="kpi-value emerald" id="kpiMonthlySavings">₹33.99 L / mo</div>
      <div class="kpi-sub" id="kpiAnnualSavings">12.64% Saved • ₹4.14 Cr / Year</div>
    </div>
    <div class="card">
      <div class="kpi-label">AI Alpha vs. Naive OA Trader</div>
      <div class="kpi-value blue" id="kpiAlpha">+₹14.67 L / mo</div>
      <div class="kpi-sub" id="kpiMix">Optimal Mix: 52.4% DAM • 17.9% RTM • 29.6% DISCOM</div>
    </div>
  </div>

  <div class="charts-grid">
    <div class="card">
      <div class="chart-title">24-Hour Demo Trace (2025-07-20): Landed OA Cost vs. DISCOM ToU Tariff & AI Effective Rate</div>
      <canvas id="hourlyChart" height="145"></canvas>
    </div>
    <div class="card">
      <div class="chart-title">Monthly Procurement Cost by Strategy (₹ Lakhs)</div>
      <canvas id="strategyChart" height="145"></canvas>
    </div>
  </div>

  <div class="card">
    <div class="chart-title">Hour-by-Hour Co-Pilot Bidding & Load Shift Schedule (Demo Day: 2025-07-20, 66 kV EHV Profile)</div>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Hour</th>
            <th>DISCOM ToU (₹/kWh)</th>
            <th>GERC Breakeven MCP (₹/kWh)</th>
            <th>AI Pred DAM [P50 / P75]</th>
            <th>Actual DAM MCP</th>
            <th>Actual RTM MCP</th>
            <th>Landed OA [DAM / RTM]</th>
            <th>Effective AI Rate</th>
            <th>Source Allocation</th>
            <th>Scheduled Load (MW)</th>
          </tr>
        </thead>
        <tbody id="scheduleBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const DATA = {data_json};
    let hourlyChartInstance = null;
    let strategyChartInstance = null;

    function renderScenario(mode) {{
      const s = DATA.scenarios[mode];
      document.getElementById('kpiDiscomRate').textContent = `₹${{s.avg_discom_rate.toFixed(2)}} / kWh`;
      document.getElementById('kpiDiscomBill').textContent = `Monthly Bill: ₹${{s.monthly_cost_discom_lakhs.toFixed(2)}} Lakhs`;
      document.getElementById('kpiAiRate').textContent = `₹${{s.avg_ai_shift_rate.toFixed(2)}} / kWh`;
      document.getElementById('kpiAiBill').textContent = `Monthly Bill: ₹${{s.monthly_cost_ai_shift_lakhs.toFixed(2)}} Lakhs`;
      document.getElementById('kpiMonthlySavings').textContent = `₹${{s.monthly_savings_ai_shift_lakhs.toFixed(2)}} L / mo`;
      document.getElementById('kpiAnnualSavings').textContent = `${{s.savings_pct_ai_shift.toFixed(2)}}% Saved • ₹${{s.annual_savings_ai_shift_crores.toFixed(2)}} Cr / Year`;
      document.getElementById('kpiAlpha').textContent = `+₹${{s.ai_alpha_over_naive_lakhs_mo.toFixed(2)}} L / mo`;
      document.getElementById('kpiMix').textContent = `Mix: ${{s.dam_energy_share_pct}}% DAM • ${{s.rtm_energy_share_pct}}% RTM • ${{s.discom_energy_share_pct}}% DISCOM`;

      if (strategyChartInstance) strategyChartInstance.destroy();
      const ctxS = document.getElementById('strategyChart').getContext('2d');
      strategyChartInstance = new Chart(ctxS, {{
        type: 'bar',
        data: {{
          labels: ['100% DISCOM', 'Naive OA Bidding', 'AI Co-Pilot (DAM+RTM)', 'AI Co-Pilot + Load Shift'],
          datasets: [{{
            label: 'Monthly Cost (₹ Lakhs)',
            data: [
              s.monthly_cost_discom_lakhs,
              s.monthly_cost_naive_lakhs,
              s.monthly_cost_ai_lakhs,
              s.monthly_cost_ai_shift_lakhs
            ],
            backgroundColor: ['#f43f5e', '#f59e0b', '#3b82f6', '#10b981'],
            borderRadius: 6
          }}]
        }},
        options: {{
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
            x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }}
          }}
        }}
      }});
    }}

    function initHourlyChartAndTable() {{
      const rows = DATA.demo_day.hourly;
      const labels = rows.map(r => `${{String(r.Hour).padStart(2, '0')}}:00`);

      const ctxH = document.getElementById('hourlyChart').getContext('2d');
      hourlyChartInstance = new Chart(ctxH, {{
        type: 'line',
        data: {{
          labels,
          datasets: [
            {{
              label: 'DISCOM ToU Rate (₹/kWh)',
              data: rows.map(r => r.DISCOM_Rate),
              borderColor: '#f43f5e',
              borderDash: [5, 5],
              tension: 0.1
            }},
            {{
              label: 'IEX DAM Landed Cost (₹/kWh)',
              data: rows.map(r => r.DAM_Landed_Cost),
              borderColor: '#94a3b8',
              tension: 0.25
            }},
            {{
              label: 'AI Co-Pilot Effective Rate (₹/kWh)',
              data: rows.map(r => r.Effective_AI_Rate),
              borderColor: '#10b981',
              backgroundColor: 'rgba(16, 185, 129, 0.12)',
              fill: true,
              borderWidth: 3,
              tension: 0.25
            }}
          ]
        }},
        options: {{
          plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }},
          scales: {{
            y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
            x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }}
          }}
        }}
      }});

      const tbody = document.getElementById('scheduleBody');
      tbody.innerHTML = rows.map(r => {{
        const isOA = r.OA_Share > 0;
        const badge = isOA
          ? `<span class="badge badge-oa">${{Math.round(r.OA_Share*100)}}% IEX OA (DAM/RTM)</span>`
          : `<span class="badge badge-discom">100% DISCOM</span>`;
        return `<tr>
          <td><strong>${{String(r.Hour).padStart(2, '0')}}:00–${{String(r.Hour+1).padStart(2, '0')}}:00</strong></td>
          <td>₹${{r.DISCOM_Rate.toFixed(2)}}</td>
          <td>₹${{r.Breakeven_MCP.toFixed(2)}}</td>
          <td>₹${{r.Pred_DAM_P50.toFixed(2)}} / ₹${{r.Pred_DAM_P75.toFixed(2)}}</td>
          <td>₹${{r.DAM_MCP.toFixed(2)}}</td>
          <td>₹${{r.RTM_MCP.toFixed(2)}}</td>
          <td>₹${{r.DAM_Landed_Cost.toFixed(2)}} / ₹${{r.RTM_Landed_Cost.toFixed(2)}}</td>
          <td class="emerald"><strong>₹${{r.Effective_AI_Rate.toFixed(2)}}</strong></td>
          <td>${{badge}}</td>
          <td><strong>${{r.Shifted_Load_MW.toFixed(2)}} MW</strong></td>
        </tr>`;
      }}).join('');
    }}

    renderScenario('66kV_Standard_OA');
    initHourlyChartAndTable();
  </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_html_path
