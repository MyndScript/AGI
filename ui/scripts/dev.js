import esbuild from 'esbuild';
import { spawn } from 'child_process';

async function start() {
  const ctx = await esbuild.context({
    entryPoints: ['./src/index.jsx'],
    bundle: true,
    outfile: './dist/bundle.js',
    sourcemap: true,
    loader: { '.js': 'jsx', '.jsx': 'jsx' },
    define: { 'process.env.NODE_ENV': '"development"' },
  });
  await ctx.watch();
  console.log('Watching for changes...');
  spawn('npx', ['serve', 'dist', '-l', '3000'], { stdio: 'inherit', shell: true });
}

start();
