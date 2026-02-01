// components/Navbar.tsx
'use client';
import { useState, useEffect } from 'react';
import { useSession, signOut } from 'next-auth/react';
import Link from 'next/link';
import { Menu, X, LogOut, Home, Camera, User, UtensilsCrossed } from 'lucide-react';
import { usePathname } from 'next/navigation';
import Image from 'next/image';

export default function Navbar() {
  const { status } = useSession();
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const toggleMenu = () => setIsOpen(!isOpen);

  const isActive = (path: string) => pathname === path;

  return (
    <nav
      className={`fixed top-0 left-0 w-full text-gray-900 shadow-lg transition-all duration-300 z-50 ${
        scrolled
          ? 'bg-white/95 backdrop-blur-xl border-b border-gray-200'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <Link
              href="/"
              className="text-xl sm:text-2xl font-bold tracking-tight hover:opacity-80 transition-all duration-300 flex items-center gap-2 group"
            >
              <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform shadow-lg">
                <UtensilsCrossed className="w-6 h-6 text-white" />
              </div>
              <span className="sm:inline bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                Smart Dietitian
              </span>
            </Link>
          </div>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center space-x-1">
            <Link
              href="/"
              className={`px-4 py-2 text-sm font-medium hover:bg-green-50 hover:scale-105 transform transition-all duration-300 rounded-lg flex items-center gap-2 ${
                isActive('/') ? 'bg-green-50 text-green-700' : 'text-gray-700'
              }`}
            >
              <Home className="w-4 h-4" />
              Home
            </Link>

            <Link
              href="/classifier"
              className={`px-4 py-2 text-sm font-medium hover:bg-green-50 hover:scale-105 transform transition-all duration-300 rounded-lg flex items-center gap-2 ${
                isActive('/classifier') ? 'bg-green-50 text-green-700' : 'text-gray-700'
              }`}
            >
              <Camera className="w-4 h-4" />
              Classifier
            </Link>

            {status === 'authenticated' ? (
              <>
                <Link
                  href="/profile"
                  className={`px-4 py-2 text-sm font-medium hover:bg-green-50 hover:scale-105 transform transition-all duration-300 rounded-lg flex items-center gap-2 ${
                    isActive('/profile') ? 'bg-green-50 text-green-700' : 'text-gray-700'
                  }`}
                >
                  <User className="w-4 h-4" />
                  Profile
                </Link>

                <button
                  onClick={() => signOut({ callbackUrl: '/' })}
                  className="px-4 py-2 text-sm font-medium hover:bg-red-50 hover:scale-105 transform transition-all duration-300 rounded-lg flex items-center gap-2 text-red-600 hover:text-red-700 ml-2"
                >
                  <LogOut className="w-4 h-4" />
                  Sign Out
                </button>
              </>
            ) : (
              <>
                <Link
                  href="/signin"
                  className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-green-600 hover:scale-105 transform transition-all duration-300 rounded-lg ml-2"
                >
                  Sign In
                </Link>
                <Link
                  href="/signup"
                  className="px-5 py-2 text-sm font-medium bg-gradient-to-r from-green-600 to-emerald-600 text-white hover:from-green-700 hover:to-emerald-700 hover:scale-105 transform transition-all duration-300 rounded-lg shadow-lg"
                >
                  Sign Up
                </Link>
              </>
            )}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={toggleMenu}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors duration-300"
              aria-label="Toggle menu"
            >
              {isOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isOpen && (
          <div className="md:hidden animate-slideDown">
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-white/95 backdrop-blur-xl rounded-lg mb-4 border border-gray-200 shadow-lg">
              <Link
                href="/"
                className={`block px-3 py-2 text-base font-medium hover:bg-green-50 rounded-lg transition-colors duration-300 flex items-center gap-2 ${
                  isActive('/') ? 'bg-green-50 text-green-700' : 'text-gray-700'
                }`}
                onClick={toggleMenu}
              >
                <Home className="w-4 h-4" />
                Home
              </Link>

              <Link
                href="/classifier"
                className={`block px-3 py-2 text-base font-medium hover:bg-green-50 rounded-lg transition-colors duration-300 flex items-center gap-2 ${
                  isActive('/classifier') ? 'bg-green-50 text-green-700' : 'text-gray-700'
                }`}
                onClick={toggleMenu}
              >
                <Camera className="w-4 h-4" />
                Classifier
              </Link>

              {status === 'authenticated' ? (
                <>
                  <Link
                    href="/profile"
                    className={`block px-3 py-2 text-base font-medium hover:bg-green-50 rounded-lg transition-colors duration-300 flex items-center gap-2 ${
                      isActive('/profile') ? 'bg-green-50 text-green-700' : 'text-gray-700'
                    }`}
                    onClick={toggleMenu}
                  >
                    <User className="w-4 h-4" />
                    Profile
                  </Link>

                  <div className="border-t border-gray-200 my-2"></div>

                  <button
                    onClick={() => {
                      signOut({ callbackUrl: '/' });
                      toggleMenu();
                    }}
                    className="block w-full text-left px-3 py-2 text-base font-medium hover:bg-red-50 rounded-lg transition-colors duration-300 flex items-center gap-2 text-red-600"
                  >
                    <LogOut className="w-4 h-4" />
                    Sign Out
                  </button>
                </>
              ) : (
                <>
                  <div className="border-t border-gray-200 my-2"></div>
                  
                  <Link
                    href="/signin"
                    className="block px-3 py-2 text-base font-medium hover:bg-gray-100 rounded-lg transition-colors duration-300 text-gray-700"
                    onClick={toggleMenu}
                  >
                    Sign In
                  </Link>
                  <Link
                    href="/signup"
                    className="block px-3 py-2 text-base font-medium bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg transition-colors duration-300 text-center"
                    onClick={toggleMenu}
                  >
                    Sign Up
                  </Link>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-slideDown {
          animation: slideDown 0.3s ease-out;
        }
      `}</style>
    </nav>
  );
}
