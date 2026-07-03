import type { Metadata, Viewport } from "next"
import { Geist_Mono, Playfair_Display } from "next/font/google"
import "./globals.css"

const playfairDisplay = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-playfair",
})

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
})

export const metadata: Metadata = {
  title: "Enhinged | Brutalist Chat Interface",
  description: "A brutalist void chat frontend for the Enhinged Hinglish model.",
}

export const viewport: Viewport = {
  themeColor: "#050505",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${playfairDisplay.variable} ${geistMono.variable}`}>
      <body className="overflow-x-hidden antialiased">
        <div className="noise-overlay" />
        {children}
      </body>
    </html>
  )
}