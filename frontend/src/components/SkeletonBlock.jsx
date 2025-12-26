import SkeletonLine from './SkeletonLine';

export default function SkeletonBlock({ lines = 3, spacing = 'space-y-3' }) {
  return (
    <div className={spacing}>
      {Array.from({ length: lines }).map((_, index) => {
        // Vary widths for last line to look more natural
        const isLastLine = index === lines - 1;
        const width = isLastLine ? 'w-3/4' : 'w-full';
        
        return <SkeletonLine key={index} width={width} />;
      })}
    </div>
  );
}
