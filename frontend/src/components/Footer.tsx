import React from 'react';
import { Link } from 'react-router-dom';
import { Facebook, Twitter, Instagram, Linkedin, Github } from 'lucide-react';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();
  
  const socialLinks = [
    { icon: <Facebook className="h-5 w-5" />, url: 'https://facebook.com/paksatalker' },
    { icon: <Twitter className="h-5 w-5" />, url: 'https://twitter.com/paksatalker' },
    { icon: <Instagram className="h-5 w-5" />, url: 'https://instagram.com/paksatalker' },
    { icon: <Linkedin className="h-5 w-5" />, url: 'https://linkedin.com/company/paksatalker' },
    { icon: <Github className="h-5 w-5" />, url: 'https://github.com/paksatalker' },
  ];

  const footerLinks = [
    {
      title: 'Product',
      links: [
        { name: 'Features', href: '/features' },
        { name: 'Pricing', href: '/pricing' },
        { name: 'Documentation', href: '/docs' },
      ],
    },
    {
      title: 'Company',
      links: [
        { name: 'About', href: '/about' },
        { name: 'Blog', href: '/blog' },
        { name: 'Careers', href: '/careers' },
      ],
    },
    {
      title: 'Legal',
      links: [
        { name: 'Privacy', href: '/privacy' },
        { name: 'Terms', href: '/terms' },
        { name: 'Cookie Policy', href: '/cookie-policy' },
      ],
    },
  ];

  return (
    <footer className="bg-gray-50 border-t border-gray-200">
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:py-16 lg:px-8">
        <div className="xl:grid xl:grid-cols-3 xl:gap-8">
          <div className="space-y-8 xl:col-span-1">
            <h3 className="text-2xl font-bold text-gray-900">PaksaTalker</h3>
            <p className="text-gray-500 text-base">
              AI-powered video generation platform that brings your content to life.
            </p>
            <div className="flex space-x-6">
              {socialLinks.map((item, index) => (
                <a
                  key={index}
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-gray-500"
                >
                  <span className="sr-only">{item.url.split('/').pop()}</span>
                  {item.icon}
                </a>
              ))}
            </div>
          </div>
          <div className="mt-12 grid grid-cols-2 gap-8 xl:mt-0 xl:col-span-2">
            <div className="md:grid md:grid-cols-3 md:gap-8">
              {footerLinks.map((section) => (
                <div key={section.title} className="mt-12 md:mt-0">
                  <h3 className="text-sm font-semibold text-gray-400 tracking-wider uppercase">
                    {section.title}
                  </h3>
                  <ul className="mt-4 space-y-4">
                    {section.links.map((link) => (
                      <li key={link.name}>
                        <Link
                          to={link.href}
                          className="text-base text-gray-500 hover:text-gray-900"
                        >
                          {link.name}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="mt-12 border-t border-gray-200 pt-8">
          <p className="text-base text-gray-400 text-center">
            &copy; {currentYear} PaksaTalker. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
