import Link from "next/link"

interface FooterLinksProps {
  authorName: string
  authorUrl: string
  photoUrl: string
}

export function FooterLinks() {
  const photoCredit: FooterLinksProps = {
    authorName: "Jan Böttinger",
    authorUrl: "https://unsplash.com/@bttngr?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText",
    photoUrl:
      "https://unsplash.com/photos/winding-road-through-a-snow-covered-forest-3m3F4-cXUKQ?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText",
  }

  return (
    <footer className="flex flex-col items-center justify-center gap-2 text-xs text-white/60">
      <div className="flex items-center gap-4">
        <Link href="/privacy" className="transition-colors hover:text-white">
          Privacy Policy
        </Link>
        <span className="text-white/30">•</span>
        <Link href="/terms" className="transition-colors hover:text-white">
          Terms & Conditions
        </Link>
      </div>
      <p className="text-white/40">
        Photo by{" "}
        <a
          href={photoCredit.authorUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="transition-colors hover:text-white"
        >
          {photoCredit.authorName}
        </a>{" "}
        on{" "}
        <a
          href={photoCredit.photoUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="transition-colors hover:text-white"
        >
          Unsplash
        </a>
      </p>
    </footer>
  )
}
