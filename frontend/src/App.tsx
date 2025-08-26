import { Routes, Route } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

// Lazy load pages for better performance
const Home = lazy(() => import('@/pages/Home'));
const Pricing = lazy(() => import('@/pages/Pricing'));
const Features = lazy(() => import('@/pages/Features'));
const Contact = lazy(() => import('@/pages/Contact'));
const Dashboard = lazy(() => import('@/pages/Dashboard'));

function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-grow">
        <Suspense fallback={<LoadingSpinner />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/pricing" element={<Pricing />} />
            <Route path="/features" element={<Features />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/dashboard/*" element={<Dashboard />} />
          </Routes>
        </Suspense>
      </main>
      <Footer />
    </div>
  );
}

export default App;
