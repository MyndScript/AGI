// Minimal esbuild build script for React
import esbuild from 'esbuild';
import { copyFileSync } from 'fs';

// UI runs on port 3000; backend ports: 8000 (Agent), 8001 (Memory), 8001 (Personality)

esbuild.build({
  entryPoints: ['./src/index.jsx'],
  bundle: true,
  outfile: './dist/bundle.js',
  minify: true,
  sourcemap: true,
  loader: { '.js': 'jsx', '.jsx': 'jsx' },
  define: { 'process.env.NODE_ENV': '"production"' },
}).then(() => {
  // Copy index.html to dist folder
  copyFileSync('./src/index.html', './dist/index.html');
  console.log('Build complete! index.html copied to dist/');
}).catch(() => process.exit(1));
