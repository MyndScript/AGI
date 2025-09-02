import esbuild from 'esbuild';
import { spawn } from 'child_process';
import { copyFileSync } from 'fs';

async function start() {
  // Copy index.html to dist folder initially
  copyFileSync('./src/index.html', './dist/index.html');
  
  const ctx = await esbuild.context({
    entryPoints: ['./src/index.jsx'],
    bundle: true,
    outfile: './dist/bundle.js',
    sourcemap: true,
    loader: { '.js': 'jsx', '.jsx': 'jsx' },
    define: { 'process.env.NODE_ENV': '"development"' },
  });
  await ctx.watch();
  console.log('Watching for changes... index.html copied to dist/');
  spawn('npx', ['serve', 'dist', '-l', '3000'], { stdio: 'inherit', shell: true });
}

start();
