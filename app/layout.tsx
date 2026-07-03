import type { Metadata, Viewport } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import "../styles/animations.css"
import "./globals.css"

const geistSans = Geist({
  subsets: ["latin"],
  variable: "--font-geist",
})

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
})

export const metadata: Metadata = {
  title: "Enhinged | Hinglish GPT Chat Interface",
  description: "A premium AI chat interface for the Enhinged Hinglish language model. Features a custom GPT-2 architecture trained on everyday conversational Hinglish.",
  openGraph: {
    title: "Enhinged | Hinglish GPT",
    description: "Experience a completely bespoke Hinglish conversational AI in a premium, glassmorphic interface.",
    type: "website",
  },
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
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`} suppressHydrationWarning>
      <body className="overflow-x-hidden antialiased bg-background text-foreground">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange
        >
          <div className="noise-overlay" />
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}