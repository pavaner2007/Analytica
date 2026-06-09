const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

async function parseResponse(response) {
  if (!response.ok) {
    let message = `Request failed with ${response.status}`
    try {
      const data = await response.json()
      message = data.detail || data.message || message
    } catch (_) {}
    throw new Error(message)
  }
  return response.json()
}

export async function loadSample() {
  const response = await fetch(`${API_BASE}/api/load-sample`, { method: 'POST' })
  return parseResponse(response)
}

export async function uploadDataset(file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData })
  return parseResponse(response)
}

export async function cleanDataset(sessionId, payload) {
  const response = await fetch(`${API_BASE}/api/clean/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseResponse(response)
}

export async function transformDataset(sessionId, payload) {
  const response = await fetch(`${API_BASE}/api/transform/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseResponse(response)
}

export async function filterDataset(sessionId, payload) {
  const response = await fetch(`${API_BASE}/api/filter/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseResponse(response)
}

export async function fetchChart(sessionId, params) {
  const search = new URLSearchParams(params)
  const response = await fetch(`${API_BASE}/api/chart/${sessionId}?${search.toString()}`)
  return parseResponse(response)
}

export async function trainModel(sessionId, payload) {
  const response = await fetch(`${API_BASE}/api/train/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseResponse(response)
}

export async function fetchAudit() {
  const response = await fetch(`${API_BASE}/api/audit?limit=100`)
  return parseResponse(response)
}

export function exportUrl(sessionId) {
  return `${API_BASE}/api/export/${sessionId}`
}
