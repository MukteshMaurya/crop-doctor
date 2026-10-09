# AI Plant Disease Detector — Frontend

A responsive, dependency-free frontend (plain HTML, CSS, and
JavaScript) for the AI Plant Disease Detector backend.

## Files

| File         | Purpose                                             |
| ------------ | --------------------------------------------------- |
| `index.html` | Page structure and UI                                |
| `styles.css` | Agricultural green theme, responsive layout          |
| `app.js`     | Upload, drag-and-drop, API call, result rendering    |
| `config.js`  | **Backend API URL — edit this to configure**         |
| `vercel.json`| Static security headers for Vercel hosting           |

## Configuration

The backend URL lives in `config.js`:

```js
window.APP_CONFIG = {
  API_BASE_URL: "http://127.0.0.1:8000", // local development
};
```

For production, set it to the deployed Render backend URL, e.g.:

```js
window.APP_CONFIG = {
  API_BASE_URL: "https://ai-plant-disease-detector-backend.onrender.com",
};
```

No other file needs to change — `app.js` reads `window.APP_CONFIG`.

## Local development

1. Start the backend (see `backend/README` or `PROJECT_PROGRESS.md`):
   ```
   cd backend
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
2. Serve this folder with any static server, e.g.:
   ```
   python -m http.server 5500
   ```
3. Open <http://localhost:5500>. The backend allows
   `http://localhost:5500` by default for development CORS.

> Serving over `http://localhost` is required for browser CORS to
> work; opening `index.html` directly via `file://` will fail.

## Deploying to Vercel

1. Push the repository to GitHub.
2. In the Vercel dashboard: **Add New → Project → import the repo**.
3. Set **Root Directory** to `frontend`.
4. Framework preset: **Other** (plain static files).
5. Leave **Build Command** empty and **Output Directory** as `.`
   (or blank).
6. Before deploying, update `config.js` with the production
   Render backend URL.
7. Deploy, then open the live URL and upload a leaf image to verify
   the request reaches Render (not localhost).

## Notes

- The frontend never contains model weights or credentials.
- File type and size (10 MB) are validated in the browser for
  convenience; the backend validation is authoritative.
- The deployed model classifies crop/plant type (17 classes), not
  disease — the UI states this clearly.
