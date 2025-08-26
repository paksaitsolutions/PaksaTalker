import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import type { PluginOption } from 'vite';
import { visualizer } from 'rollup-plugin-visualizer';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load environment variables
  const env = loadEnv(mode, process.cwd(), '');
  const isProduction = mode === 'production';
  
  return {
    // Base public path when served in development or production
    base: isProduction ? '/static/' : '/',
    
    // Plugins
    plugins: [
      react({
        // Use React 17+ JSX runtime
        jsxRuntime: 'automatic',
        // Babel configuration
        babel: {
          plugins: [
            ['@babel/plugin-proposal-decorators', { legacy: true }],
            ['@babel/plugin-proposal-class-properties', { loose: true }],
          ],
        },
      }),
      // Visualize bundle size in production
      isProduction ? visualizer({
        open: true,
        gzipSize: true,
        brotliSize: true,
      }) as PluginOption : null,
    ].filter(Boolean),
    
    // Resolve options
    resolve: {
      alias: {
        '@': resolve(__dirname, './src'),
        '~': __dirname,
      },
      extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json'],
    },
    
    // Environment variables
    define: {
      'process.env': {},
      __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
    },
    
    // Development server configuration
    server: {
      port: 3000,
      strictPort: true,
      host: '0.0.0.0',
      open: !process.env.CI,
      cors: true,
      fs: {
        // Allow serving files from one level up from the package root
        allow: ['..'],
      },
      proxy: {
        '/api': {
          target: env.VITE_API_BASE_URL || 'http://localhost:5000',
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/api/, '/api/v1'),
          configure: (proxy, _options) => {
            proxy.on('error', (err, _req, _res) => {
              console.error('Proxy error:', err);
            });
            proxy.on('proxyReq', (proxyReq, req, _res) => {
              console.log('Sending Request to the Target:', req.method, req.url);
            });
            proxy.on('proxyRes', (proxyRes, req, _res) => {
              console.log('Received Response from the Target:', proxyRes.statusCode, req.url);
            });
          },
        },
      },
    },
    
    // Build configuration
    build: {
      outDir: '../static',
      emptyOutDir: true,
      sourcemap: isProduction ? 'hidden' : true,
      minify: isProduction ? 'terser' : false,
      cssMinify: isProduction,
      target: 'esnext',
      modulePreload: {
        polyfill: false,
      },
      chunkSizeWarningLimit: 1000,
      reportCompressedSize: false,
      rollupOptions: {
        output: {
          manualChunks: {
            react: ['react', 'react-dom', 'react-router-dom'],
            vendor: ['axios', 'date-fns', '@heroicons/react', 'lucide-react'],
          },
          entryFileNames: 'assets/[name].[hash].js',
          chunkFileNames: 'assets/[name].[hash].js',
          assetFileNames: 'assets/[name].[hash].[ext]',
        },
        onwarn(warning, warn) {
          // Ignore certain warnings
          if (warning.code === 'MODULE_LEVEL_DIRECTIVE') {
            return;
          }
          warn(warning);
        },
      },
      terserOptions: {
        compress: {
          drop_console: isProduction,
          drop_debugger: isProduction,
        },
        format: {
          comments: false,
        },
      },
    },
    
    // CSS configuration
    css: {
      devSourcemap: true,
      modules: {
        localsConvention: 'camelCaseOnly',
      },
      preprocessorOptions: {
        scss: {
          additionalData: `@import "@/styles/_variables.scss";`,
        },
      },
    },
    
    // Optimize dependencies
    optimizeDeps: {
      include: ['react', 'react-dom', 'react-router-dom'],
      exclude: ['@babel/plugin-transform-runtime'],
      esbuildOptions: {
        // Enable esbuild's tree shaking
        treeShaking: true,
      },
    },
    
    // Log level
    logLevel: isProduction ? 'info' : 'warn',
  };
});
