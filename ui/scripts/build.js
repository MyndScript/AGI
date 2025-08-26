// Minimal esbuild build script for React
import esbuild from 'esbuild';

esbuild.build({
  entryPoints: ['./src/index.jsx'],
  bundle: true,
  outfile: './dist/bundle.js',
  minify: true,
  sourcemap: true,
  loader: { '.js': 'jsx', '.jsx': 'jsx' },
  define: { 'process.env.NODE_ENV': '"production"' },
}).catch(() => process.exit(1));
