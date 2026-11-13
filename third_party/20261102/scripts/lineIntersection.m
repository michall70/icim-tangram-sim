function [x, y] = lineIntersection(x1, y1, x2, y2, x3, y3, x4, y4)
    x = NaN;
    y = NaN;

    % Direction vectors of the two lines
    dx1 = x2 - x1;
    dy1 = y2 - y1;
    dx2 = x4 - x3;
    dy2 = y4 - y3;

    % Reject identical endpoints
    if (dx1 == 0 && dy1 == 0) || (dx2 == 0 && dy2 == 0)
        return;
    end

    denominator = dx1 * dy2 - dy1 * dx2;

    % Parallel or coincident lines have no unique intersection
    if denominator == 0
        return;
    end

    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denominator;

    x = x1 + t * dx1;
    y = y1 + t * dy1;
end