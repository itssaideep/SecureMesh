import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { Shield, AlertTriangle, Activity, Server } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const MOCK_DATA = {
  totalAttacks: 1245,
  highRisk: 42,
  activeAttackers: 8
};

function App() {
  const [stats, setStats] = useState(MOCK_DATA);
  const [logs, setLogs] = useState([]);
  
  const chartData = {
    labels: ['10:00', '11:00', '12:00', '13:00', '14:00', '15:00'],
    datasets: [
      {
        label: 'SSH Attempts',
        data: [65, 59, 80, 81, 56, 120],
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      },
      {
        label: 'Telnet Attempts',
        data: [28, 48, 40, 19, 86, 27],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      }
    ],
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans">
      <header className="mb-8">
        <h1 className="text-3xl font-bold flex items-center text-blue-400">
          <Shield className="mr-3" size={32} />
          AI-Powered IoT Honeypot Dashboard
        </h1>
        <p className="text-slate-400 mt-2">Real-time threat monitoring and AI analysis</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg flex flex-col items-center">
          <Activity size={32} className="text-blue-400 mb-2" />
          <h2 className="text-slate-400 text-lg">Total Attacks</h2>
          <p className="text-4xl font-bold">{stats.totalAttacks}</p>
        </div>
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg flex flex-col items-center">
          <AlertTriangle size={32} className="text-red-400 mb-2" />
          <h2 className="text-slate-400 text-lg">High Risk Alerts</h2>
          <p className="text-4xl font-bold text-red-400">{stats.highRisk}</p>
        </div>
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg flex flex-col items-center">
          <Server size={32} className="text-green-400 mb-2" />
          <h2 className="text-slate-400 text-lg">Active Attackers</h2>
          <p className="text-4xl font-bold text-green-400">{stats.activeAttackers}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
          <h3 className="text-xl font-semibold mb-4 text-blue-300">Attack Timeline</h3>
          <Line options={{ responsive: true }} data={chartData} />
        </div>

        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg flex flex-col">
          <h3 className="text-xl font-semibold mb-4 text-blue-300">AI Analysis Panel</h3>
          <div className="flex-grow flex items-center justify-center border-2 border-dashed border-slate-600 rounded-lg bg-slate-800/50 p-4 text-center">
            <p className="text-slate-400 italic">"Possible malware deployment attempt detected. Attacker downloaded executable shell script. Risk Level: High."</p>
          </div>
        </div>
      </div>
      
      <div className="mt-8 bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
         <h3 className="text-xl font-semibold mb-4 text-blue-300">Live Alert Status (ESP8266 Simulator)</h3>
         <div className="flex items-center space-x-4">
            <div className={`w-6 h-6 rounded-full ${stats.highRisk > 0 ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}></div>
            <span className="text-lg">{stats.highRisk > 0 ? 'ATTACK DETECTED (Alert GUI Active)' : 'System Normal'}</span>
         </div>
      </div>
    </div>
  );
}

export default App;
